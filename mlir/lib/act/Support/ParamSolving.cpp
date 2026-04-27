#include "act/Support/ParamSolving.h"
#include "act/Support/SymbolicExpr.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <utility>

#define DEBUG_TYPE "param-solving"

using namespace mlir;
using namespace mlir::act;

using llvm::dbgs;

namespace {
struct AccessBufferInfo {
  StringAttr name;
  BufferTypeInterface type;
  AccessRole role = AccessRole::Read;
};

struct AccessShapeConstraint {
  unsigned operandIdx = 0;
  SmallVector<int64_t> sourceShape;
  const SemanticInputBinding *binding = nullptr;
};
} // namespace

static SmallVector<AccessBufferInfo> getAccessBufferInfos(DefineOp defineOp,
                                                          ModuleOp module) {
  SmallVector<AccessBufferInfo> infos;

  auto appendInfo = [&](FlatSymbolRefAttr ref, AccessRole role) {
    auto buffer =
        SymbolTable::lookupNearestSymbolFrom<DeclareBufferOp>(module, ref);
    assert(buffer && "buffer symbol should be verified before param solving");
    infos.push_back({ref.getAttr(), buffer.getBufferType(), role});
  };

  for (auto source : defineOp.getSources().getAsRange<FlatSymbolRefAttr>())
    appendInfo(source, AccessRole::Read);
  for (auto dest : defineOp.getDestinations().getAsRange<FlatSymbolRefAttr>())
    appendInfo(dest, AccessRole::Write);

  return infos;
}

static void markParam(DenseMap<unsigned, AddrParamKind> &kinds, unsigned idx,
                      AddrParamKind kind) {
  auto it = kinds.find(idx);
  if (it == kinds.end()) {
    kinds[idx] = kind;
    return;
  }
  if (it->second != kind)
    it->second = AddrParamKind::Mixed;
}

static LogicalResult markExprParams(OpFoldResult value, AddrParamKind kind,
                                    DenseMap<unsigned, AddrParamKind> &kinds,
                                    Operation *op) {
  auto expr = buildSymExpr(value);
  if (failed(expr))
    return op->emitError()
           << "failed to build symbolic expression for addr parameter "
              "classification";

  DenseSet<unsigned> params;
  expr->collectParams(params);
  for (unsigned idx : params)
    markParam(kinds, idx, kind);

  return success();
}

static LogicalResult
classifyAddrParams(DefineOp defineOp,
                   DenseMap<unsigned, AddrParamKind> &kinds) {
  Block &addrBlock = defineOp.getAccessBlock();

  for (Operation &op : addrBlock) {
    if (auto strided = dyn_cast<StridedOp>(&op)) {
      MLIRContext *ctx = strided.getContext();
      auto basis =
          getMixedValues(strided.getStaticBasis(), strided.getBasis(), ctx);
      auto counts =
          getMixedValues(strided.getStaticCounts(), strided.getCounts(), ctx);
      auto strides =
          getMixedValues(strided.getStaticStrides(), strided.getStrides(), ctx);

      for (OpFoldResult value : basis)
        if (failed(markExprParams(value, AddrParamKind::Offset, kinds, &op)))
          return failure();
      for (OpFoldResult value : strides)
        if (failed(markExprParams(value, AddrParamKind::Offset, kinds, &op)))
          return failure();
      for (OpFoldResult value : counts)
        if (failed(markExprParams(value, AddrParamKind::Shape, kinds, &op)))
          return failure();
      continue;
    }

    if (auto expand = dyn_cast<ExpandShapeOp>(&op)) {
      auto outputShape =
          getMixedValues(expand.getStaticOutputShape(), expand.getOutputShape(),
                         expand.getContext());
      for (OpFoldResult value : outputShape)
        if (failed(markExprParams(value, AddrParamKind::Shape, kinds, &op)))
          return failure();
    }
  }

  return success();
}

static FailureOr<SymShape> buildSymExprs(ArrayRef<OpFoldResult> values,
                                         Operation *op, StringRef name) {
  SymShape exprs;
  for (OpFoldResult value : values) {
    auto expr = buildSymExpr(value);
    if (failed(expr))
      return op->emitError()
             << "failed to build symbolic " << name << " expression";
    exprs.push_back(std::move(*expr));
  }
  return exprs;
}

static FailureOr<SymbolicRegion> generateStridedStorageRegion(StridedOp op) {
  MLIRContext *ctx = op.getContext();
  auto basis = getMixedValues(op.getStaticBasis(), op.getBasis(), ctx);
  auto counts = getMixedValues(op.getStaticCounts(), op.getCounts(), ctx);
  auto strides = getMixedValues(op.getStaticStrides(), op.getStrides(), ctx);

  auto basisExprs = buildSymExprs(basis, op, "basis");
  auto countExprs = buildSymExprs(counts, op, "count");
  auto strideExprs = buildSymExprs(strides, op, "stride");
  if (failed(basisExprs) || failed(countExprs) || failed(strideExprs))
    return failure();

  SymbolicRegion region;
  region.basis = std::move(*basisExprs);
  region.counts = std::move(*countExprs);
  region.strides = std::move(*strideExprs);
  return region;
}

static FailureOr<SymbolicRegion> generateStorageRegion(Operation *accessOp) {
  assert(accessOp && "expected access pattern op");

  if (auto strided = dyn_cast<StridedOp>(accessOp))
    return generateStridedStorageRegion(strided);

  if (auto expand = dyn_cast<ExpandShapeOp>(accessOp)) {
    Operation *sourceOp = expand.getSource().getDefiningOp();
    if (!sourceOp)
      return expand.emitError() << "source access pattern has no defining op";
    return generateStorageRegion(sourceOp);
  }

  if (auto collapse = dyn_cast<CollapseShapeOp>(accessOp)) {
    Operation *sourceOp = collapse.getSource().getDefiningOp();
    if (!sourceOp)
      return collapse.emitError() << "source access pattern has no defining op";
    return generateStorageRegion(sourceOp);
  }

  if (auto transpose = dyn_cast<TransposeOp>(accessOp)) {
    Operation *sourceOp = transpose.getSource().getDefiningOp();
    if (!sourceOp)
      return transpose.emitError()
             << "source access pattern has no defining op";
    return generateStorageRegion(sourceOp);
  }

  if (isa<TiledOp>(accessOp))
    return accessOp->emitError()
           << "symbolic storage extraction for act.tiled is not supported yet";

  return accessOp->emitError()
         << "unsupported access pattern op for symbolic storage extraction";
}

FailureOr<InstructionParamModel>
act::buildInstructionParamModel(DefineOp defineOp, ModuleOp module) {
  InstructionParamModel model;
  model.defineOp = defineOp;

  SmallVector<AccessBufferInfo> bufferInfos =
      getAccessBufferInfos(defineOp, module);
  Operation *terminator = defineOp.getAccessBlock().getTerminator();
  assert(terminator && "access region should have a terminator");
  assert(terminator->getNumOperands() == bufferInfos.size() &&
         "access yield operands should match src+dst buffers");

  for (auto [idx, operand] : llvm::enumerate(terminator->getOperands())) {
    Operation *accessOp = operand.getDefiningOp();
    if (!accessOp)
      return defineOp.emitError()
             << "access yield operand " << idx << " has no defining op";

    AccessBufferInfo &info = bufferInfos[idx];
    auto shape = generateShapeExpr(accessOp, info.type);
    if (failed(shape))
      return failure();

    auto storage = generateStorageRegion(accessOp);
    if (failed(storage))
      return failure();

    SymbolicAccess access;
    access.operandIdx = static_cast<unsigned>(idx);
    access.bufferName = info.name;
    access.bufferType = info.type;
    access.role = info.role;
    access.storage = std::move(*storage);
    access.visibleShape = std::move(*shape);
    model.accesses.push_back(std::move(access));
  }

  if (failed(classifyAddrParams(defineOp, model.paramKinds)))
    return failure();

  return model;
}

static std::string intShapeToString(ArrayRef<int64_t> shape) {
  std::string result = "[";
  for (auto [idx, dim] : llvm::enumerate(shape)) {
    if (idx != 0)
      result += ", ";
    result += std::to_string(dim);
  }
  result += "]";
  return result;
}

static const SymbolicAccess *findAccess(InstructionParamModel &model,
                                        unsigned operandIdx) {
  for (SymbolicAccess &access : model.accesses)
    if (access.operandIdx == operandIdx)
      return &access;
  return nullptr;
}

static FailureOr<SmallVector<int64_t>> getStaticTensorShape(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return failure();
  if (!type.hasStaticShape())
    return failure();

  SmallVector<int64_t> shape;
  llvm::append_range(shape, type.getShape());
  return shape;
}

static FailureOr<SmallVector<AccessShapeConstraint, 4>>
collectSourceShapeConstraints(SemanticGraphNode &node) {
  SmallVector<AccessShapeConstraint, 4> constraints;
  for (SemanticInputBinding &binding : node.inputBindings) {
    auto shape = getStaticTensorShape(binding.value);
    if (failed(shape)) {
      LLVM_DEBUG(dbgs() << "  [reject] bound value for operand "
                        << binding.accessOperandIdx
                        << " is not a ranked static tensor\n");
      return failure();
    }

    constraints.push_back(
        {binding.accessOperandIdx, std::move(*shape), &binding});
  }
  return constraints;
}

static std::optional<int64_t>
getSolvedExprValue(const SymExpr &expr,
                   DenseMap<unsigned, int64_t> &solvedParams) {
  switch (expr.kind) {
  case SymExpr::Kind::Constant:
    return expr.value;
  case SymExpr::Kind::Param: {
    auto it = solvedParams.find(expr.paramIdx);
    if (it == solvedParams.end())
      return std::nullopt;
    return it->second;
  }
  case SymExpr::Kind::Add: {
    assert(expr.lhs && expr.rhs && "expected binary expression operands");
    auto lhs = getSolvedExprValue(*expr.lhs, solvedParams);
    auto rhs = getSolvedExprValue(*expr.rhs, solvedParams);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs + *rhs;
  }
  case SymExpr::Kind::Mul: {
    assert(expr.lhs && expr.rhs && "expected binary expression operands");
    auto lhs = getSolvedExprValue(*expr.lhs, solvedParams);
    auto rhs = getSolvedExprValue(*expr.rhs, solvedParams);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs * *rhs;
  }
  }
  llvm_unreachable("unknown symbolic expression kind");
}

static bool isShapeParam(unsigned idx, InstructionParamModel &model) {
  auto it = model.paramKinds.find(idx);
  assert(it != model.paramKinds.end() &&
         "shape expression param should be classified");
  return it->second != AddrParamKind::Offset;
}

static LogicalResult bindParam(unsigned idx, int64_t value,
                               InstructionParamModel &model,
                               DenseMap<unsigned, int64_t> &solvedParams) {
  if (!isShapeParam(idx, model)) {
    LLVM_DEBUG(dbgs() << "  [reject] offset param p" << idx
                      << " appears in a shape constraint\n");
    return failure();
  }

  auto it = solvedParams.find(idx);
  if (it == solvedParams.end()) {
    solvedParams[idx] = value;
    return success();
  }
  if (it->second == value)
    return success();

  LLVM_DEBUG(dbgs() << "  [reject] p" << idx << " constrained to " << it->second
                    << " and " << value << "\n");
  return failure();
}

static LogicalResult solveExpr(const SymExpr &expr, int64_t expected,
                               InstructionParamModel &model,
                               DenseMap<unsigned, int64_t> &solvedParams) {
  switch (expr.kind) {
  case SymExpr::Kind::Constant:
    if (expr.value == expected)
      return success();
    LLVM_DEBUG(dbgs() << "  [reject] constant shape mismatch: "
                      << expr.toString() << " != " << expected << "\n");
    return failure();
  case SymExpr::Kind::Param:
    return bindParam(expr.paramIdx, expected, model, solvedParams);
  case SymExpr::Kind::Add: {
    assert(expr.lhs && expr.rhs && "expected binary expression operands");
    auto lhs = getSolvedExprValue(*expr.lhs, solvedParams);
    if (lhs)
      return solveExpr(*expr.rhs, expected - *lhs, model, solvedParams);

    auto rhs = getSolvedExprValue(*expr.rhs, solvedParams);
    if (rhs)
      return solveExpr(*expr.lhs, expected - *rhs, model, solvedParams);

    LLVM_DEBUG(dbgs() << "  [reject] cannot solve additive expression "
                      << expr.toString() << " == " << expected << "\n");
    return failure();
  }
  case SymExpr::Kind::Mul: {
    assert(expr.lhs && expr.rhs && "expected binary expression operands");
    auto lhs = getSolvedExprValue(*expr.lhs, solvedParams);
    if (lhs) {
      if (*lhs == 0 || expected % *lhs != 0) {
        LLVM_DEBUG(dbgs() << "  [reject] cannot divide " << expected
                          << " by factor " << *lhs << " in " << expr.toString()
                          << "\n");
        return failure();
      }
      return solveExpr(*expr.rhs, expected / *lhs, model, solvedParams);
    }

    auto rhs = getSolvedExprValue(*expr.rhs, solvedParams);
    if (rhs) {
      if (*rhs == 0 || expected % *rhs != 0) {
        LLVM_DEBUG(dbgs() << "  [reject] cannot divide " << expected
                          << " by factor " << *rhs << " in " << expr.toString()
                          << "\n");
        return failure();
      }
      return solveExpr(*expr.lhs, expected / *rhs, model, solvedParams);
    }

    LLVM_DEBUG(dbgs() << "  [reject] cannot solve multiplicative expression "
                      << expr.toString() << " == " << expected << "\n");
    return failure();
  }
  }
  llvm_unreachable("unknown symbolic expression kind");
}

static LogicalResult
solveShapeConstraints(InstructionParamModel &model,
                      ArrayRef<AccessShapeConstraint> constraints,
                      DenseMap<unsigned, int64_t> &solvedParams) {
  DenseSet<unsigned> constrainedOperands;

  for (const AccessShapeConstraint &constraint : constraints) {
    const SymbolicAccess *access = findAccess(model, constraint.operandIdx);
    if (!access) {
      LLVM_DEBUG(dbgs() << "  [reject] no symbolic access for operand "
                        << constraint.operandIdx << "\n");
      return failure();
    }

    constrainedOperands.insert(constraint.operandIdx);
    if (access->visibleShape.size() != constraint.sourceShape.size()) {
      LLVM_DEBUG(dbgs() << "  [reject] operand " << constraint.operandIdx
                        << " rank mismatch: symbolic "
                        << symShapeToString(access->visibleShape) << " source "
                        << intShapeToString(constraint.sourceShape) << "\n");
      return failure();
    }

    for (auto [symbolic, concrete] :
         llvm::zip(access->visibleShape, constraint.sourceShape))
      if (failed(solveExpr(symbolic, concrete, model, solvedParams)))
        return failure();
  }

  for (SymbolicAccess &access : model.accesses) {
    if (constrainedOperands.contains(access.operandIdx))
      continue;
    LLVM_DEBUG(dbgs() << "  [reject] no source shape constraint for operand "
                      << access.operandIdx << "\n");
    return failure();
  }

  return success();
}

static StringRef accessRoleToString(AccessRole role) {
  switch (role) {
  case AccessRole::Read:
    return "read";
  case AccessRole::Write:
    return "write";
  case AccessRole::ReadWrite:
    return "readwrite";
  }
  llvm_unreachable("unknown access role");
}

static void dumpAccessModel(InstructionParamModel &model) {
  dbgs() << "  access model:\n";
  for (SymbolicAccess &access : model.accesses) {
    dbgs() << "    operand " << access.operandIdx << " "
           << accessRoleToString(access.role) << " @"
           << access.bufferName.getValue() << " storage=basis"
           << symShapeToString(access.storage.basis) << " counts"
           << symShapeToString(access.storage.counts) << " strides"
           << symShapeToString(access.storage.strides)
           << " visible=" << symShapeToString(access.visibleShape) << "\n";
  }
}

static void dumpParamSolution(ParamSolution &solution,
                              ArrayRef<AccessShapeConstraint> constraints) {
  DefineOp defineOp = solution.model.defineOp;
  dbgs() << "Solving @" << defineOp.getSymName() << "\n";
  dbgs() << "  source ops:";
  for (SemanticIdentity &identity : solution.node.sourceOps)
    dbgs() << " " << identity.op->getName();
  dbgs() << "\n";
  dumpAccessModel(solution.model);

  for (const AccessShapeConstraint &constraint : constraints) {
    const SymbolicAccess *access =
        findAccess(solution.model, constraint.operandIdx);
    assert(access && "constraint should reference a valid symbolic access");
    dbgs() << "  operand " << constraint.operandIdx << ": symbolic "
           << symShapeToString(access->visibleShape) << " source "
           << intShapeToString(constraint.sourceShape) << " from pattern "
           << constraint.binding->patternNodeId << ":"
           << constraint.binding->patternOperandId << "\n";
  }

  dbgs() << "  solved params:";
  bool hasSolved = false;
  for (unsigned idx = 0; idx < defineOp.getAccessBlock().getNumArguments();
       ++idx) {
    auto it = solution.solvedParams.find(idx);
    if (it == solution.solvedParams.end())
      continue;
    dbgs() << " p" << idx << "=" << it->second;
    hasSolved = true;
  }
  if (!hasSolved)
    dbgs() << " <none>";
  dbgs() << "\n";

  dbgs() << "  unsolved offset params:";
  bool hasOffset = false;
  for (unsigned idx = 0; idx < defineOp.getAccessBlock().getNumArguments();
       ++idx) {
    auto kind = solution.model.paramKinds.find(idx);
    if (kind == solution.model.paramKinds.end() ||
        kind->second != AddrParamKind::Offset ||
        solution.solvedParams.contains(idx))
      continue;
    dbgs() << " p" << idx;
    hasOffset = true;
  }
  if (!hasOffset)
    dbgs() << " <none>";
  dbgs() << "\n";
}

FailureOr<GraphParamSolution> act::runParamSolving(SemanticGraph &graph) {
  LLVM_DEBUG(dbgs() << "=== Running parameter solving for semantic graph @"
                    << graph.func.getSymName() << " ===\n");

  ModuleOp module = graph.func->getParentOfType<ModuleOp>();
  assert(module && "semantic graph function should belong to a module");

  GraphParamSolution solutions;
  solutions.reserve(graph.nodes.size());
  for (SemanticGraphNode &node : graph.nodes) {
    auto model = buildInstructionParamModel(node.pattern.instruction, module);
    if (failed(model))
      return failure();

    auto constraints = collectSourceShapeConstraints(node);
    if (failed(constraints))
      return failure();

    DenseMap<unsigned, int64_t> solvedParams;
    if (failed(solveShapeConstraints(*model, *constraints, solvedParams)))
      return failure();

    solutions.emplace_back(node, std::move(*model));
    ParamSolution &solution = solutions.back();
    solution.solvedParams = std::move(solvedParams);
    solution.isValid = true;

    LLVM_DEBUG(dumpParamSolution(solution, *constraints));
  }

  return std::move(solutions);
}
