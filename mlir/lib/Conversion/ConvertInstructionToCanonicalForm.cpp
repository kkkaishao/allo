#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h"

namespace mlir::allo {
#define GEN_PASS_DEF_CONVERTINSTRUCTIONTOCANONICALFORMPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {
struct BlockAccessPattern {
  SmallVector<int64_t, 4> extents;
  SmallVector<int64_t, 4> offsets;
  SmallVector<int64_t, 4> strides;
};
} // namespace

static FailureOr<BlockAccessPattern> extractBlockAccessPattern(AffineMap map,
                                                               IntegerSet set) {
  // extract the block access pattern
  // Case 1: affine_map<(d0, d1) -> (2 + d0, 3 + 2 * d1)> => offsets = [2, 3],
  // strides = [1, 2] Case 2: affine_map<(d0, d1) -> (d0, d1)> => offsets = [0,
  // 0], strides = [1, 1]

  // Case 1: affine_set<(d0, d1) : (d0 >= 0, d0 + 1 <= 10, d1 >= 0, d1 + 1 <=
  // 20)> => extents = [10, 20]
  unsigned numDims = map.getNumDims();
  if (set.getNumDims() != numDims || map.getNumResults() == 0 ||
      map.getNumSymbols() != 0 || set.getNumSymbols() != 0)
    return failure();

  struct LinearExpr {
    SmallVector<int64_t, 4> coeffs;
    int64_t constant = 0;
  };

  auto parseLinearExpr = [&](AffineExpr expr,
                             auto &&self) -> FailureOr<LinearExpr> {
    if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
      LinearExpr out;
      out.coeffs.assign(numDims, 0);
      out.constant = cst.getValue();
      return out;
    }
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      LinearExpr out;
      out.coeffs.assign(numDims, 0);
      out.coeffs[dim.getPosition()] = 1;
      return out;
    }
    if (isa<AffineSymbolExpr>(expr))
      return failure();

    auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!bin)
      return failure();

    AffineExprKind kind = bin.getKind();
    if (kind == AffineExprKind::Add) {
      auto lhsOr = self(bin.getLHS(), self);
      auto rhsOr = self(bin.getRHS(), self);
      if (failed(lhsOr) || failed(rhsOr))
        return failure();
      LinearExpr out;
      out.coeffs.resize(numDims, 0);
      for (unsigned i = 0; i < numDims; ++i)
        out.coeffs[i] = (*lhsOr).coeffs[i] + (*rhsOr).coeffs[i];
      out.constant = (*lhsOr).constant + (*rhsOr).constant;
      return out;
    }

    if (kind == AffineExprKind::Mul) {
      auto rhsCst = dyn_cast<AffineConstantExpr>(bin.getRHS());
      auto lhsCst = dyn_cast<AffineConstantExpr>(bin.getLHS());
      if (rhsCst) {
        auto lhsOr = self(bin.getLHS(), self);
        if (failed(lhsOr))
          return failure();
        int64_t factor = rhsCst.getValue();
        LinearExpr out = *lhsOr;
        for (int64_t &coeff : out.coeffs)
          coeff *= factor;
        out.constant *= factor;
        return out;
      }
      if (lhsCst) {
        auto rhsOr = self(bin.getRHS(), self);
        if (failed(rhsOr))
          return failure();
        int64_t factor = lhsCst.getValue();
        LinearExpr out = *rhsOr;
        for (int64_t &coeff : out.coeffs)
          coeff *= factor;
        out.constant *= factor;
        return out;
      }
    }

    return failure();
  };

  auto floorDiv = [](int64_t a, int64_t b) -> int64_t {
    assert(b > 0 && "expected positive divisor");
    int64_t q = a / b;
    int64_t r = a % b;
    if (r < 0)
      --q;
    return q;
  };
  auto ceilDiv = [&](int64_t a, int64_t b) -> int64_t {
    assert(b > 0 && "expected positive divisor");
    int64_t q = a / b;
    int64_t r = a % b;
    if (r > 0)
      ++q;
    return q;
  };

  SmallVector<int64_t, 4> lowerBounds(numDims,
                                      std::numeric_limits<int64_t>::min());
  SmallVector<int64_t, 4> upperBounds(numDims,
                                      std::numeric_limits<int64_t>::max());
  SmallVector<bool, 4> hasLower(numDims, false), hasUpper(numDims, false);

  for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
    auto linOr = parseLinearExpr(set.getConstraint(i), parseLinearExpr);
    if (failed(linOr))
      return failure();
    const LinearExpr &lin = *linOr;

    int nonZeroPos = -1;
    int64_t coeff = 0;
    for (unsigned d = 0; d < numDims; ++d) {
      if (lin.coeffs[d] == 0)
        continue;
      if (nonZeroPos != -1)
        return failure();
      nonZeroPos = static_cast<int>(d);
      coeff = lin.coeffs[d];
    }
    if (nonZeroPos < 0)
      return failure();

    unsigned dim = static_cast<unsigned>(nonZeroPos);
    if (set.isEq(i)) {
      if (coeff == 0)
        return failure();
      int64_t rhs = -lin.constant;
      if (rhs % coeff != 0)
        return failure();
      int64_t v = rhs / coeff;
      lowerBounds[dim] = std::max(lowerBounds[dim], v);
      upperBounds[dim] = std::min(upperBounds[dim], v);
      hasLower[dim] = true;
      hasUpper[dim] = true;
      continue;
    }

    // IntegerSet inequalities are of the form: coeff * d + c >= 0.
    if (coeff > 0) {
      int64_t lb = ceilDiv(-lin.constant, coeff);
      lowerBounds[dim] = std::max(lowerBounds[dim], lb);
      hasLower[dim] = true;
    } else if (coeff < 0) {
      int64_t ub = floorDiv(lin.constant, -coeff);
      upperBounds[dim] = std::min(upperBounds[dim], ub);
      hasUpper[dim] = true;
    } else {
      return failure();
    }
  }

  struct MapAccessDesc {
    unsigned dim;
    int64_t baseOffset;
    int64_t stride;
  };
  SmallVector<MapAccessDesc, 4> accesses;
  accesses.reserve(map.getNumResults());
  for (AffineExpr resultExpr : map.getResults()) {
    auto linOr = parseLinearExpr(resultExpr, parseLinearExpr);
    if (failed(linOr))
      return failure();
    const LinearExpr &lin = *linOr;

    int nonZeroPos = -1;
    int64_t stride = 0;
    for (unsigned d = 0; d < numDims; ++d) {
      if (lin.coeffs[d] == 0)
        continue;
      if (nonZeroPos != -1)
        return failure();
      nonZeroPos = static_cast<int>(d);
      stride = lin.coeffs[d];
    }
    if (nonZeroPos < 0 || stride <= 0)
      return failure();

    accesses.push_back(
        {static_cast<unsigned>(nonZeroPos), lin.constant, stride});
  }

  BlockAccessPattern pattern;
  pattern.extents.reserve(accesses.size());
  pattern.offsets.reserve(accesses.size());
  pattern.strides.reserve(accesses.size());

  for (const MapAccessDesc &access : accesses) {
    unsigned dim = access.dim;
    if (!hasLower[dim] || !hasUpper[dim] || lowerBounds[dim] > upperBounds[dim])
      return failure();

    int64_t extent = upperBounds[dim] - lowerBounds[dim] + 1;
    int64_t offset = access.baseOffset + access.stride * lowerBounds[dim];
    pattern.extents.push_back(extent);
    pattern.offsets.push_back(offset);
    pattern.strides.push_back(access.stride);
  }
  return pattern;
}

static Value generateBlockLoadAccess(ConversionPatternRewriter &rewriter,
                                     Location loc, Value operand,
                                     BlockAccessPattern &pattern) {
  Type type = operand.getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    auto resType =
        RankedTensorType::get(pattern.extents, tensorTy.getElementType());
    Value extracted = tensor::ExtractSliceOp::create(
        rewriter, loc, resType, operand, {}, {}, {}, pattern.offsets,
        pattern.extents, pattern.strides);
    return extracted;
  }
  if (isa<MemRefType>(type)) {
    Value subview =
        memref::SubViewOp::create(rewriter, loc, operand, pattern.offsets,
                                  pattern.extents, pattern.strides);
    return subview;
  }
  llvm_unreachable("unexpected type for load access");
}

static std::optional<Value>
generateBlockStoreAccess(ConversionPatternRewriter &rewriter, Location loc,
                         Value src, Value dst, BlockAccessPattern &pattern) {
  Type type = dst.getType();
  if (isa<RankedTensorType>(type)) {
    Value inserted = tensor::InsertSliceOp::create(
        rewriter, loc, dst.getType(), src, dst, {}, {}, {}, pattern.offsets,
        pattern.extents, pattern.strides);
    return inserted;
  }
  if (isa<MemRefType>(type)) {
    // memref is memory semantics, not value semantics
    // no need to generate a store op, just return the subview
    return std::nullopt;
  }
  llvm_unreachable("unexpected type for store access");
}

static AffineMap replaceMapSymbolsToConstants(AffineMap map,
                                              ArrayRef<int64_t> values) {
  SmallVector<AffineExpr, 8> dimRepls;
  SmallVector<AffineExpr, 8> symRepls;
  dimRepls.reserve(map.getNumDims());
  symRepls.reserve(map.getNumSymbols());

  MLIRContext *ctx = map.getContext();
  for (unsigned i = 0; i < map.getNumDims(); ++i)
    dimRepls.push_back(getAffineDimExpr(i, ctx));
  for (unsigned i = 0; i < map.getNumSymbols(); ++i)
    symRepls.push_back(getAffineConstantExpr(values[i], ctx));

  SmallVector<AffineExpr, 8> newResults;
  for (auto res : map.getResults())
    newResults.push_back(res.replaceDimsAndSymbols(dimRepls, symRepls));

  auto newMap = AffineMap::get(map.getNumDims(), 0, newResults, ctx);
  newMap = simplifyAffineMap(newMap);
  newMap = compressUnusedSymbols(newMap);
  return newMap;
}

static IntegerSet replaceSetSymbolsToConstants(IntegerSet set,
                                               ArrayRef<int64_t> values) {
  SmallVector<AffineExpr, 8> dimRepls;
  SmallVector<AffineExpr, 8> symRepls;
  MLIRContext *ctx = set.getContext();
  for (unsigned i = 0; i < set.getNumDims(); ++i)
    dimRepls.push_back(getAffineDimExpr(i, ctx));
  for (unsigned i = 0; i < set.getNumSymbols(); ++i)
    symRepls.push_back(getAffineConstantExpr(values[i], ctx));

  SmallVector<AffineExpr, 8> newConstraints;
  for (auto cst : set.getConstraints())
    newConstraints.push_back(cst.replaceDimsAndSymbols(dimRepls, symRepls));

  auto newSet =
      IntegerSet::get(set.getNumDims(), 0, newConstraints, set.getEqFlags());
  affine::simplifyIntegerSet(newSet);
  return newSet;
}

static void
composeMapsAndSets(InstructionDefineOp defineOp,
                   llvm::SmallDenseMap<StringRef, int64_t> &symValMap,
                   SmallVectorImpl<AffineMap> &maps,
                   SmallVectorImpl<IntegerSet> &sets) {
  auto indexMaps =
      llvm::to_vector<4>(defineOp.getIndexMaps().getAsRange<AffineMapAttr>());
  SmallVector<int64_t, 8> indexSymbolValues;
  for (auto sym : defineOp.getIndexSymbols().getAsRange<StringAttr>()) {
    auto name = sym.getValue();
    assert(symValMap.contains(name));
    indexSymbolValues.push_back(symValMap[name]);
  }
  for (auto map : indexMaps) {
    maps.push_back(
        replaceMapSymbolsToConstants(map.getValue(), indexSymbolValues));
  }
  auto indexSets =
      llvm::to_vector<4>(defineOp.getIndexSets().getAsRange<IntegerSetAttr>());
  SmallVector<int64_t, 8> domainSymbolValues;
  for (auto sym : defineOp.getDomainSymbols().getAsRange<StringAttr>()) {
    auto name = sym.getValue();
    assert(symValMap.contains(name));
    domainSymbolValues.push_back(symValMap[name]);
  }
  for (auto set : indexSets) {
    sets.push_back(
        replaceSetSymbolsToConstants(set.getValue(), domainSymbolValues));
  }
}

static bool isCastLikeArithOp(Operation *op) {
  return isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::ExtFOp,
             arith::TruncFOp, arith::BitcastOp, arith::IndexCastOp>(op);
}

namespace {
struct SemanticBlockInstantiator {
  SemanticBlockInstantiator(RewriterBase &b, IRMapping &m, Location loc)
      : b(b), mapping(m), loc(loc) {}

  FailureOr<SmallVector<Value, 4>> rebuild(Block &block) const {
    SmallVector<Value, 4> results;
    for (Operation &op : block.getOperations()) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        if (failed(rebuildLinalgStructuredOp(linalgOp)))
          return failure();
        continue;
      }
      SmallVector<Value, 4> newOperands;
      for (Value operand : op.getOperands())
        newOperands.push_back(mapping.lookupOrDefault(operand));
      if (isa<arith::ArithDialect, math::MathDialect>(op.getDialect())) {
        assert(op.getNumResults() == 1 &&
               "expected single result for arith/math op");
        Type newType = newOperands.front().getType();
        Type castType = op.getResult(0).getType();
        if (isCastLikeArithOp(&op)) {
          if (auto tensorTy = dyn_cast<RankedTensorType>(newType)) {
            auto origTensor = cast<RankedTensorType>(castType);
            newType = RankedTensorType::get(tensorTy.getShape(),
                                            origTensor.getElementType(),
                                            tensorTy.getEncoding());
          } else if (auto memrefTy = dyn_cast<MemRefType>(newType)) {
            auto origMemref = cast<MemRefType>(castType);
            newType = MemRefType::get(
                memrefTy.getShape(), origMemref.getElementType(),
                memrefTy.getLayout(), memrefTy.getMemorySpace());
          } else {
            newType = castType;
          }
        }
        rebuildTrivialOp(&op, newOperands, newType, op.getAttrs());
        continue;
      }
      if (isa<linalg::YieldOp>(op)) {
        rebuildTrivialOp(&op, newOperands, {}, op.getAttrs());
        continue;
      }
      if (isa<InstructionYieldOp>(op)) {
        results = std::move(newOperands);
        break;
      }
      return op.emitError()
             << "unsupported operation in semantic block: " << op.getName();
    }
    return results;
  }

private:
  RewriterBase &b;
  IRMapping &mapping;
  Location &loc; // location of emit op

  void rebuildTrivialOp(Operation *op, ArrayRef<Value> newOperands,
                        ArrayRef<Type> newTypes,
                        ArrayRef<NamedAttribute> attrs) const {
    OperationState state(loc, op->getName());
    state.addOperands(newOperands);
    state.addTypes(newTypes);
    state.addAttributes(attrs);
    Operation *rebuilt = b.create(state);
    for (auto [origResult, newResult] :
         llvm::zip_equal(op->getResults(), rebuilt->getResults()))
      mapping.map(origResult, newResult);
  }

  LogicalResult rebuildLinalgStructuredOp(linalg::LinalgOp op) const {
    // create a dummy version to analyze the result shapes
    auto reifyOp = cast<ReifyRankedShapedTypeOpInterface>(
        b.clone(*op.getOperation(), mapping));
    // infer new return types
    ReifiedRankedShapedTypeDims shapes;
    shapes.reserve(op->getNumResults());
    if (failed(reifyOp.reifyResultShapes(b, shapes)))
      return op.emitError() << "failed to infer result shapes for linalg op";

    SmallVector<Type, 4> resTypes;
    for (auto [shape, origType] :
         llvm::zip_equal(shapes, op->getResultTypes())) {
      SmallVector<int64_t, 4> dims;
      for (OpFoldResult ofr : shape) {
        if (auto attr = dyn_cast<Attribute>(ofr)) {
          if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
            dims.push_back(intAttr.getInt());
            continue;
          }
        }
        return op.emitError() << "expected static shape for linalg op result";
      }
      auto origTensor = cast<RankedTensorType>(origType);
      resTypes.push_back(RankedTensorType::get(
          dims, origTensor.getElementType(), origTensor.getEncoding()));
    }

    OperationState state(loc, op->getName());
    state.addOperands(reifyOp->getOperands());
    state.addTypes(resTypes);
    state.addAttributes(op->getAttrs());
    Region *region = state.addRegion();
    Block &body = region->emplaceBlock();
    // map block arguments
    SmallVector<Location, 4> argLocs(reifyOp->getNumOperands(), loc);
    auto blockArgs =
        body.addArguments(op.getBlock()->getArgumentTypes(), argLocs);
    for (auto [origArg, newArg] :
         llvm::zip_equal(op.getBlock()->getArguments(), blockArgs))
      mapping.map(origArg, newArg);
    Operation *rebuilt = b.create(state);
    b.setInsertionPointToStart(&body);
    if (failed(rebuild(*op.getBlock())))
      return failure();
    b.setInsertionPointAfter(rebuilt);
    // map the results
    for (auto [origResult, newResult] :
         llvm::zip_equal(op->getResults(), rebuilt->getResults()))
      mapping.map(origResult, newResult);
    return success();
  }
};
} // namespace

namespace {
struct EmitOpLowering : public OpConversionPattern<InstructionEmitOp> {
  EmitOpLowering(MLIRContext *context, SymbolTableCollection &symbolTables)
      : OpConversionPattern(context), symbolTables(symbolTables) {}

  LogicalResult
  matchAndRewrite(InstructionEmitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto defineOp = symbolTables.lookupNearestSymbolFrom<InstructionDefineOp>(
        op, op.getInstructionAttr());
    if (!defineOp)
      return failure();

    // get symbol values
    llvm::SmallDenseMap<StringRef, int64_t> symValMap;
    for (auto sym : op.getSymbolValues()) {
      auto name = sym.getName().getValue();
      int64_t val = cast<IntegerAttr>(sym.getValue()).getInt();
      symValMap[name] = val;
    }
    SmallVector<AffineMap, 4> composedMaps;
    SmallVector<IntegerSet, 4> composedSets;
    composeMapsAndSets(defineOp, symValMap, composedMaps, composedSets);

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    IRMapping mapping;
    Block *semantics = defineOp.getSemantics();
    // generate load operations for each operand of the instruction emit op
    // based on the composed affine maps and sets
    // map all operands, including both sources and destinations
    auto newOperands = adaptor.getOperands();
    for (unsigned i = 0; i < newOperands.size(); ++i) {
      AffineMap map = composedMaps[i];
      IntegerSet set = composedSets[i];
      auto patternOr = extractBlockAccessPattern(map, set);
      if (failed(patternOr))
        return failure();
      Value extracted =
          generateBlockLoadAccess(rewriter, loc, newOperands[i], *patternOr);
      mapping.map(semantics->getArgument(i), extracted);
    }
    // instantiate the semantic block
    SemanticBlockInstantiator instantiator(rewriter, mapping, op.getLoc());
    auto yieldedOr = instantiator.rebuild(*semantics);
    if (failed(yieldedOr))
      return failure();
    auto &yielded = *yieldedOr;
    // generate store operations for each destination
    auto newDsts = op.getDestinations();
    unsigned nSources = adaptor.getSources().size();
    SmallVector<Value, 4> finalResults;
    for (unsigned i = 0; i < newDsts.size(); ++i) {
      AffineMap map = composedMaps[nSources + i];
      IntegerSet set = composedSets[nSources + i];
      auto patternOr = extractBlockAccessPattern(map, set);
      if (failed(patternOr))
        return failure();
      if (auto insertedOr = generateBlockStoreAccess(rewriter, loc, yielded[i],
                                                     newDsts[i], *patternOr))
        finalResults.push_back(*insertedOr);
    }
    // replace the emit op with the final results
    rewriter.replaceOp(op, finalResults);
    return success();
  }

private:
  SymbolTableCollection &symbolTables;
};
} // namespace

namespace {
struct ConvertInstructionToCanonicalFormPass
    : public allo::impl::ConvertInstructionToCanonicalFormPassBase<
          ConvertInstructionToCanonicalFormPass> {
  ConvertInstructionToCanonicalFormPass() = default;
  explicit ConvertInstructionToCanonicalFormPass(
      const ConvertInstructionToCanonicalFormPassOptions &options) {
    stripDefinitions = options.stripDefinitions;
  }

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    SymbolTableCollection symbolTables;
    ConversionConfig config;
    config.allowPatternRollback = false;

    RewritePatternSet patterns(ctx);
    patterns.add<EmitOpLowering>(ctx, symbolTables);

    ConversionTarget target(*ctx);
    target.addIllegalOp<InstructionEmitOp>();
    // primitive op set
    target.addLegalOp<
#define GET_LINALG_OPS
#include "AllowedOps.h.inc"
        >();
    target.addLegalOp<
#define GET_TENSOR_OPS
#include "AllowedOps.h.inc"
        >();
    target.addLegalDialect<arith::ArithDialect, math::MathDialect,
                           func::FuncDialect>();

    if (failed(
            applyPartialConversion(mod, target, std::move(patterns), config)))
      signalPassFailure();

    // run canonicalization after conversion
    patterns.clear();
    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      signalPassFailure();

    if (stripDefinitions) {
      SmallVector<Operation *, 4> toErase;
      mod.walk(
          [&](InstructionDefineOp defineOp) { toErase.push_back(defineOp); });
      for (Operation *op : toErase)
        op->erase();
    }
  }

private:
  bool stripDefinitions = false;
};
} // namespace
