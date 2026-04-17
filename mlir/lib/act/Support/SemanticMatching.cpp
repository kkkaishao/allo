#include "act/Support/SemanticMatching.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <tuple>

#define DEBUG_TYPE "semantic-matching"

using namespace mlir;
using namespace mlir::act;

namespace {

struct RelayoutChain {
  Value source;
  SemanticEdgeTransformChain transforms;
  Operation *unsupported = nullptr;
};

struct MatchState {
  const InstructionGraph &pattern;
  const ProgramGraph &program;
  SmallVector<unsigned, 4> patternToSourceNodeIdx;
  SmallVector<unsigned, 4> sourceToPatternNodeIdx;
  SmallVector<bool, 4> mappedPatternNodes;
  SmallVector<bool, 8> usedSourceNodes;
};

struct MaterializedOutputInfo {
  Value value;
  SemanticEdgeTransformChain transforms;
};

} // namespace

static void printSliceSpec(raw_ostream &os, const StaticSliceSpec &slice) {
  auto printVec = [&](StringRef label, ArrayRef<int64_t> values) {
    os << " " << label << "=[";
    for (auto [idx, value] : llvm::enumerate(values)) {
      if (idx)
        os << ",";
      os << value;
    }
    os << "]";
  };
  printVec("offsets", slice.offsets);
  printVec("sizes", slice.sizes);
  printVec("strides", slice.strides);
}

static void printTransform(raw_ostream &os, const SemanticEdgeTransform &tx) {
  switch (tx.kind) {
  case SemanticEdgeTransformKind::Transpose:
    os << "transpose[";
    for (auto [idx, value] : llvm::enumerate(tx.permutation)) {
      if (idx)
        os << ",";
      os << value;
    }
    os << "]";
    return;
  case SemanticEdgeTransformKind::ExtractSlice:
    os << "extract_slice";
    assert(tx.sliceSpec && "extract_slice requires static slice spec");
    printSliceSpec(os, *tx.sliceSpec);
    return;
  case SemanticEdgeTransformKind::InsertSlice:
    os << "insert_slice";
    assert(tx.sliceSpec && "insert_slice requires static slice spec");
    printSliceSpec(os, *tx.sliceSpec);
    return;
  }
  llvm_unreachable("unknown semantic edge transform");
}

static void printTransformChain(raw_ostream &os,
                                ArrayRef<SemanticEdgeTransform> transforms) {
  if (transforms.empty()) {
    os << "identity";
    return;
  }
  for (auto [idx, tx] : llvm::enumerate(transforms)) {
    if (idx)
      os << " -> ";
    printTransform(os, tx);
  }
}

static void printValueRef(raw_ostream &os, Value value) {
  if (!value) {
    os << "<null>";
    return;
  }
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    os << "%arg" << arg.getArgNumber();
    return;
  }
  auto result = cast<OpResult>(value);
  os << "%" << result.getOwner()->getName().getStringRef() << "."
     << result.getResultNumber();
}

static void printFingerprint(raw_ostream &os, const SemanticFingerprint &fp) {
  switch (fp.kind) {
  case SemanticFingerprint::Named:
    os << fp.opName;
    return;
  case SemanticFingerprint::Generic:
    os << "linalg.generic";
    return;
  case SemanticFingerprint::Identity:
    os << "identity";
    return;
  case SemanticFingerprint::Opaque:
    os << "opaque(" << fp.opName << ")";
    return;
  }
  llvm_unreachable("unknown fingerprint kind");
}

static StringRef getDefineName(DefineOp defineOp) {
  return defineOp.getSymName();
}

static StringRef getFuncName(func::FuncOp funcOp) { return funcOp.getName(); }

static std::optional<StaticSliceSpec>
getStaticSliceSpec(tensor::ExtractSliceOp sliceOp) {
  StaticSliceSpec spec;
  auto staticOffsets = sliceOp.getStaticOffsets();
  auto staticSizes = sliceOp.getStaticSizes();
  auto staticStrides = sliceOp.getStaticStrides();
  if (llvm::any_of(staticOffsets,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticSizes,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticStrides,
                   [](int64_t v) { return v == ShapedType::kDynamic; })) {
    return std::nullopt;
  }
  spec.offsets.assign(staticOffsets.begin(), staticOffsets.end());
  spec.sizes.assign(staticSizes.begin(), staticSizes.end());
  spec.strides.assign(staticStrides.begin(), staticStrides.end());
  return spec;
}

static std::optional<StaticSliceSpec>
getStaticSliceSpec(tensor::InsertSliceOp sliceOp) {
  StaticSliceSpec spec;
  auto staticOffsets = sliceOp.getStaticOffsets();
  auto staticSizes = sliceOp.getStaticSizes();
  auto staticStrides = sliceOp.getStaticStrides();
  if (llvm::any_of(staticOffsets,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticSizes,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticStrides,
                   [](int64_t v) { return v == ShapedType::kDynamic; })) {
    return std::nullopt;
  }
  spec.offsets.assign(staticOffsets.begin(), staticOffsets.end());
  spec.sizes.assign(staticSizes.begin(), staticSizes.end());
  spec.strides.assign(staticStrides.begin(), staticStrides.end());
  return spec;
}

static bool isAnyLayoutOp(Operation *op) {
  return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, tensor::InsertSliceOp,
             linalg::TransposeOp>(op);
}

static bool isAlwaysNonSemanticLayoutOp(Operation *op) {
  return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, tensor::InsertSliceOp>(op);
}

static bool isPotentialSemanticOp(Operation *op) {
  if (isa<linalg::FillOp>(op) || isAlwaysNonSemanticLayoutOp(op))
    return false;
  return isa<linalg::LinalgOp, linalg::SoftmaxOp>(op);
}

static unsigned getFingerprintSpecificity(const SemanticFingerprint &fp) {
  switch (fp.kind) {
  case SemanticFingerprint::Generic:
    return 3;
  case SemanticFingerprint::Named:
    return 2;
  default:
    return 0;
  }
}

static DenseSet<Operation *> findPreTransforms(Block &block) {
  DenseSet<Operation *> preTransforms;
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : block) {
      if (preTransforms.contains(&op) || !isAnyLayoutOp(&op))
        continue;
      bool allOperandsFromBoundary =
          llvm::all_of(op.getOperands(), [&](Value v) {
            if (isa<BlockArgument>(v))
              return true;
            if (auto *defOp = v.getDefiningOp())
              return preTransforms.contains(defOp) ||
                     isa<arith::ConstantOp>(defOp);
            return false;
          });
      if (allOperandsFromBoundary) {
        preTransforms.insert(&op);
        changed = true;
      }
    }
  }
  return preTransforms;
}

static DenseSet<Operation *> findPostTransforms(Block &block) {
  DenseSet<Operation *> postTransforms;
  Operation *yieldOp = block.getTerminator();

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : block) {
      if (postTransforms.contains(&op) || !isAnyLayoutOp(&op))
        continue;
      bool allUsersAtBoundary =
          llvm::all_of(op.getResults(), [&](Value result) {
            return llvm::all_of(result.getUsers(), [&](Operation *user) {
              return user == yieldOp || postTransforms.contains(user);
            });
          });
      if (allUsersAtBoundary) {
        postTransforms.insert(&op);
        changed = true;
      }
    }
  }
  return postTransforms;
}

static DenseSet<Operation *> collectBoundaryLayoutOps(Block &block) {
  auto preTransforms = findPreTransforms(block);
  auto postTransforms = findPostTransforms(block);
  preTransforms.insert(postTransforms.begin(), postTransforms.end());
  return preTransforms;
}

static LogicalResult validateComputeRegion(DefineOp defineOp) {
  Block &block = defineOp.getSemanticsBlock();
  auto preTransforms = findPreTransforms(block);
  auto postTransforms = findPostTransforms(block);

  bool hasErrors = false;
  for (Operation *op : preTransforms) {
    defineOp.emitError()
        << "compute region of @" << defineOp.getSymName()
        << " contains boundary layout op '" << op->getName()
        << "' that should be in the addr region (pre-transform)";
    hasErrors = true;
  }
  for (Operation *op : postTransforms) {
    defineOp.emitError()
        << "compute region of @" << defineOp.getSymName()
        << " contains boundary layout op '" << op->getName()
        << "' that should be in the addr region (post-transform)";
    hasErrors = true;
  }
  return success(!hasErrors);
}

static bool isSemanticNodeOp(Operation *op,
                             const DenseSet<Operation *> &boundaryLayoutOps) {
  if (!isPotentialSemanticOp(op))
    return false;
  if (isAlwaysNonSemanticLayoutOp(op))
    return false;
  if (isa<linalg::TransposeOp>(op) && boundaryLayoutOps.contains(op))
    return false;
  return true;
}

/// Walk value-def chains through layout-only relayouts until reaching either a
/// semantic node result or an external value. The returned transform chain is
/// ordered from the boundary-visible value toward the consumer.
static RelayoutChain resolveValue(Value value,
                                  const DenseSet<Operation *> &semanticNodes) {
  Value current = value;
  SemanticEdgeTransformChain transforms;

  while (Operation *defOp = current.getDefiningOp()) {
    if (semanticNodes.contains(defOp))
      break;
    if (auto transposeOp = dyn_cast<linalg::TransposeOp>(defOp)) {
      SemanticEdgeTransform tx;
      tx.kind = SemanticEdgeTransformKind::Transpose;
      tx.permutation.assign(transposeOp.getPermutation().begin(),
                            transposeOp.getPermutation().end());
      transforms.push_back(std::move(tx));
      current = transposeOp.getInput();
      continue;
    }
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      auto sliceSpec = getStaticSliceSpec(extractSliceOp);
      if (!sliceSpec)
        return {current, {}, extractSliceOp};
      SemanticEdgeTransform tx;
      tx.kind = SemanticEdgeTransformKind::ExtractSlice;
      tx.sliceSpec = *sliceSpec;
      transforms.push_back(std::move(tx));
      current = extractSliceOp.getSource();
      continue;
    }
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(defOp)) {
      auto sliceSpec = getStaticSliceSpec(insertSliceOp);
      if (!sliceSpec)
        return {current, {}, insertSliceOp};
      SemanticEdgeTransform tx;
      tx.kind = SemanticEdgeTransformKind::InsertSlice;
      tx.sliceSpec = *sliceSpec;
      transforms.push_back(std::move(tx));
      current = insertSliceOp.getSource();
      continue;
    }
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(defOp)) {
      current = expandOp.getSrc();
      continue;
    }
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(defOp)) {
      current = collapseOp.getSrc();
      continue;
    }
    break;
  }

  return {current, std::move(transforms), nullptr};
}

/// Trace forward from a produced compute value to the unique external
/// materialized value that escapes the selected node, if such a value exists.
static FailureOr<std::optional<MaterializedOutputInfo>>
resolveUniqueMaterializedOutput(Value producedValue,
                                const DenseSet<Operation *> &semanticNodes) {
  struct WorkItem {
    Value value;
    SemanticEdgeTransformChain forwardTransforms;
  };

  SmallVector<WorkItem, 4> worklist;
  worklist.push_back({producedValue, {}});
  std::optional<MaterializedOutputInfo> materialized;

  while (!worklist.empty()) {
    WorkItem item = std::move(worklist.pop_back_val());
    bool advanced = false;
    bool sawNonSemanticUse = false;

    for (Operation *user : item.value.getUsers()) {
      if (semanticNodes.contains(user))
        continue;
      sawNonSemanticUse = true;

      if (auto transposeOp = dyn_cast<linalg::TransposeOp>(user)) {
        WorkItem next = item;
        SemanticEdgeTransform tx;
        tx.kind = SemanticEdgeTransformKind::Transpose;
        tx.permutation.assign(transposeOp.getPermutation().begin(),
                              transposeOp.getPermutation().end());
        next.forwardTransforms.push_back(std::move(tx));
        next.value = transposeOp->getResult(0);
        worklist.push_back(std::move(next));
        advanced = true;
        continue;
      }
      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
        auto sliceSpec = getStaticSliceSpec(extractSliceOp);
        if (!sliceSpec)
          return user->emitError()
                 << "dynamic layout transform is not supported by semantic "
                    "matching output binding construction";
        WorkItem next = item;
        SemanticEdgeTransform tx;
        tx.kind = SemanticEdgeTransformKind::ExtractSlice;
        tx.sliceSpec = *sliceSpec;
        next.forwardTransforms.push_back(std::move(tx));
        next.value = extractSliceOp.getResult();
        worklist.push_back(std::move(next));
        advanced = true;
        continue;
      }
      if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(user)) {
        auto sliceSpec = getStaticSliceSpec(insertSliceOp);
        if (!sliceSpec)
          return user->emitError()
                 << "dynamic layout transform is not supported by semantic "
                    "matching output binding construction";
        WorkItem next = item;
        SemanticEdgeTransform tx;
        tx.kind = SemanticEdgeTransformKind::InsertSlice;
        tx.sliceSpec = *sliceSpec;
        next.forwardTransforms.push_back(std::move(tx));
        next.value = insertSliceOp.getResult();
        worklist.push_back(std::move(next));
        advanced = true;
        continue;
      }
      if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(user)) {
        worklist.push_back({expandOp.getResult(), item.forwardTransforms});
        advanced = true;
        continue;
      }
      if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(user)) {
        worklist.push_back({collapseOp.getResult(), item.forwardTransforms});
        advanced = true;
        continue;
      }
    }

    if (advanced || !sawNonSemanticUse)
      continue;

    MaterializedOutputInfo candidate;
    candidate.value = item.value;
    candidate.transforms.assign(item.forwardTransforms.rbegin(),
                                item.forwardTransforms.rend());

    if (!materialized) {
      materialized = std::move(candidate);
      continue;
    }

    if (materialized->value == candidate.value &&
        materialized->transforms == candidate.transforms)
      continue;

    return producedValue.getDefiningOp()->emitError()
           << "selected node output escapes through multiple external "
              "materializations; Phase C1 requires a unique writeback target";
  }

  return materialized;
}

static llvm::hash_code hashGenericBody(Block *body) {
  llvm::SmallDenseMap<Value, unsigned, 16> valueIds;
  for (BlockArgument arg : body->getArguments())
    valueIds[arg] = arg.getArgNumber();

  unsigned nextValueId = body->getNumArguments();
  llvm::hash_code hash = llvm::hash_combine(body->getNumArguments());
  for (Operation &op : *body) {
    if (isa<linalg::YieldOp>(op))
      continue;
    SmallVector<unsigned, 8> operandIds;
    operandIds.reserve(op.getNumOperands());
    for (Value operand : op.getOperands()) {
      auto it = valueIds.find(operand);
      operandIds.push_back(it == valueIds.end() ? GraphEdge::kNullIdx
                                                : it->second);
    }
    llvm::hash_code opHash = llvm::hash_combine(
        op.getName().getStringRef(),
        llvm::hash_combine_range(operandIds.begin(), operandIds.end()));
    if (auto cst = dyn_cast<arith::ConstantOp>(&op))
      opHash = llvm::hash_combine(opHash, cst.getValue());
    hash = llvm::hash_combine(hash, opHash);
    for (Value result : op.getResults())
      valueIds[result] = nextValueId++;
  }
  return hash;
}

static bool genericBodiesMatch(Block *lhs, Block *rhs) {
  if (lhs->getNumArguments() != rhs->getNumArguments())
    return false;

  llvm::SmallDenseMap<Value, unsigned, 16> lhsIds;
  llvm::SmallDenseMap<Value, unsigned, 16> rhsIds;
  for (auto [idx, arg] : llvm::enumerate(lhs->getArguments()))
    lhsIds[arg] = idx;
  for (auto [idx, arg] : llvm::enumerate(rhs->getArguments()))
    rhsIds[arg] = idx;

  auto lhsIt = lhs->begin();
  auto rhsIt = rhs->begin();
  unsigned nextId = lhs->getNumArguments();
  for (; lhsIt != lhs->end() && rhsIt != rhs->end(); ++lhsIt, ++rhsIt) {
    if (isa<linalg::YieldOp>(*lhsIt) || isa<linalg::YieldOp>(*rhsIt))
      return isa<linalg::YieldOp>(*lhsIt) && isa<linalg::YieldOp>(*rhsIt);

    if (lhsIt->getName() != rhsIt->getName() ||
        lhsIt->getNumOperands() != rhsIt->getNumOperands() ||
        lhsIt->getNumResults() != rhsIt->getNumResults())
      return false;

    for (unsigned i = 0; i < lhsIt->getNumOperands(); ++i) {
      if (lhsIds.lookup(lhsIt->getOperand(i)) !=
          rhsIds.lookup(rhsIt->getOperand(i)))
        return false;
    }

    if (auto lhsConst = dyn_cast<arith::ConstantOp>(&*lhsIt)) {
      auto rhsConst = dyn_cast<arith::ConstantOp>(&*rhsIt);
      if (!rhsConst || lhsConst.getValue() != rhsConst.getValue())
        return false;
    }

    for (auto [lhsResult, rhsResult] :
         llvm::zip(lhsIt->getResults(), rhsIt->getResults())) {
      lhsIds[lhsResult] = nextId;
      rhsIds[rhsResult] = nextId;
      ++nextId;
    }
  }
  return lhsIt == lhs->end() && rhsIt == rhs->end();
}

llvm::hash_code SemanticFingerprint::hash() const {
  llvm::hash_code hash =
      llvm::hash_combine(kind, opName, numInputs, numOutputs);
  if (kind == Generic) {
    hash = llvm::hash_combine(
        hash,
        llvm::hash_combine_range(indexingMaps.begin(), indexingMaps.end()),
        llvm::hash_combine_range(iteratorTypes.begin(), iteratorTypes.end()),
        body ? hashGenericBody(body) : llvm::hash_combine(0u));
  }
  return hash;
}

bool SemanticFingerprint::matches(const SemanticFingerprint &other) const {
  if (kind != other.kind || opName != other.opName ||
      numInputs != other.numInputs || numOutputs != other.numOutputs)
    return false;
  if (kind != Generic)
    return true;
  return indexingMaps == other.indexingMaps &&
         iteratorTypes == other.iteratorTypes && body && other.body &&
         genericBodiesMatch(body, other.body);
}

SemanticFingerprint::SemanticFingerprint(Operation *op) {
  opName = op->getName().getStringRef();
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    llvm::append_range(indexingMaps, linalgOp.getIndexingMapsArray());
    llvm::append_range(iteratorTypes, linalgOp.getIteratorTypesArray());
    body = linalgOp.getBlock();
    numInputs = linalgOp.getNumDpsInputs();
    numOutputs = linalgOp.getNumDpsInits();
    kind = isa<linalg::GenericOp>(linalgOp) ? Generic : Named;
    return;
  }
  if (isa<linalg::SoftmaxOp>(op)) {
    numInputs = op->getNumOperands();
    numOutputs = op->getNumResults();
    kind = Named;
    return;
  }
  numInputs = op->getNumOperands();
  numOutputs = op->getNumResults();
  kind = Opaque;
}

static unsigned selectAnchorNode(const GraphBase &graph) {
  assert(!graph.nodes.empty() && "cannot select anchor from empty graph");
  unsigned bestIdx = 0;
  auto bestKey = std::tuple<unsigned, unsigned, int>(
      getFingerprintSpecificity(graph.nodes[0].fp),
      graph.nodes[0].inEdgeIdxs.size() + graph.nodes[0].outEdgeIdxs.size(), 0);
  for (unsigned idx = 1; idx < graph.nodes.size(); ++idx) {
    const GraphNode &node = graph.nodes[idx];
    auto key = std::tuple<unsigned, unsigned, int>(
        getFingerprintSpecificity(node.fp),
        node.inEdgeIdxs.size() + node.outEdgeIdxs.size(),
        -static_cast<int>(idx));
    if (key > bestKey) {
      bestIdx = idx;
      bestKey = key;
    }
  }
  return bestIdx;
}

static FailureOr<unsigned> getSemanticOpLoopRank(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    return linalgOp.getNumLoops();
  if (auto softmaxOp = dyn_cast<linalg::SoftmaxOp>(op))
    return softmaxOp.getInputOperandRank();
  return op->emitError() << "semantic op does not expose an iteration domain";
}

static FailureOr<unsigned> selectDomainNode(DefineOp defineOp,
                                            const GraphBase &graph) {
  assert(!graph.nodes.empty() && "cannot select domain from empty graph");

  unsigned bestIdx = 0;
  unsigned bestRank = 0;
  bool hasBest = false;
  bool ambiguous = false;

  for (unsigned idx = 0; idx < graph.nodes.size(); ++idx) {
    auto rankOr = getSemanticOpLoopRank(graph.nodes[idx].op);
    if (failed(rankOr)) {
      return defineOp.emitError()
             << "semantic op '" << graph.nodes[idx].op->getName()
             << "' does not expose an iteration domain";
    }
    unsigned rank = *rankOr;
    if (!hasBest || rank > bestRank) {
      bestIdx = idx;
      bestRank = rank;
      hasBest = true;
      ambiguous = false;
      continue;
    }

    if (rank == bestRank)
      ambiguous = true;
  }

  if (ambiguous) {
    return defineOp.emitError()
           << "compute region of @" << defineOp.getSymName()
           << " has multiple semantic ops with maximal iteration rank "
           << bestRank << "; cannot derive a unique domain carrier";
  }

  return bestIdx;
}

static unsigned countAdjacentMappedNeighbors(const InstructionGraph &pattern,
                                             const MatchState &state,
                                             unsigned patternNodeIdx) {
  unsigned count = 0;
  const GraphNode &node = pattern.nodes[patternNodeIdx];
  for (unsigned edgeIdx : node.inEdgeIdxs) {
    if (state.mappedPatternNodes[pattern.edges[edgeIdx].producerNodeIdx])
      ++count;
  }
  for (unsigned edgeIdx : node.outEdgeIdxs) {
    if (state.mappedPatternNodes[pattern.edges[edgeIdx].consumerNodeIdx])
      ++count;
  }
  return count;
}

/// Pick the next pattern node to extend from the current partial match. This
/// mirrors instruction selectors: grow from the anchor through the most
/// constrained frontier node first.
static unsigned chooseNextPatternNode(const MatchState &state) {
  unsigned bestIdx = GraphEdge::kNullIdx;
  auto bestKey = std::tuple<unsigned, unsigned, unsigned>(0, 0, 0);
  for (unsigned idx = 0; idx < state.pattern.nodes.size(); ++idx) {
    if (state.mappedPatternNodes[idx])
      continue;
    unsigned mappedNeighbors =
        countAdjacentMappedNeighbors(state.pattern, state, idx);
    const GraphNode &node = state.pattern.nodes[idx];
    auto key = std::tuple<unsigned, unsigned, unsigned>(
        mappedNeighbors, getFingerprintSpecificity(node.fp),
        node.inEdgeIdxs.size() + node.outEdgeIdxs.size());
    if (bestIdx == GraphEdge::kNullIdx || key > bestKey) {
      bestIdx = idx;
      bestKey = key;
    }
  }
  return bestIdx;
}

static bool transformsMatch(ArrayRef<SemanticEdgeTransform> lhs,
                            ArrayRef<SemanticEdgeTransform> rhs) {
  auto sliceSpecsMatch = [](const std::optional<StaticSliceSpec> &lhsSpec,
                            const std::optional<StaticSliceSpec> &rhsSpec) {
    if (lhsSpec.has_value() != rhsSpec.has_value())
      return false;
    if (!lhsSpec)
      return true;
    return lhsSpec->offsets == rhsSpec->offsets &&
           lhsSpec->sizes == rhsSpec->sizes &&
           lhsSpec->strides == rhsSpec->strides;
  };
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsTx, rhsTx] : llvm::zip(lhs, rhs)) {
    if (lhsTx.kind != rhsTx.kind || lhsTx.permutation != rhsTx.permutation ||
        !sliceSpecsMatch(lhsTx.sliceSpec, rhsTx.sliceSpec))
      return false;
  }
  return true;
}

const GraphEdge *GraphBase::findEdge(unsigned producerNodeIdx,
                                     unsigned consumerNodeIdx,
                                     unsigned producerResultIdx,
                                     unsigned consumerOperandIdx) const {
  if (producerNodeIdx >= nodes.size())
    return nullptr;
  for (unsigned edgeIdx : nodes[producerNodeIdx].outEdgeIdxs) {
    const GraphEdge &edge = edges[edgeIdx];
    if (edge.consumerNodeIdx == consumerNodeIdx &&
        edge.producerResultIdx == producerResultIdx &&
        edge.consumerOperandIdx == consumerOperandIdx)
      return &edge;
  }
  return nullptr;
}

bool GraphBase::hasAnyEdgeBetween(unsigned lhsNodeIdx,
                                  unsigned rhsNodeIdx) const {
  if (lhsNodeIdx >= nodes.size() || rhsNodeIdx >= nodes.size())
    return false;
  for (unsigned edgeIdx : nodes[lhsNodeIdx].outEdgeIdxs) {
    if (edges[edgeIdx].consumerNodeIdx == rhsNodeIdx)
      return true;
  }
  for (unsigned edgeIdx : nodes[rhsNodeIdx].outEdgeIdxs) {
    if (edges[edgeIdx].consumerNodeIdx == lhsNodeIdx)
      return true;
  }
  return false;
}

std::optional<unsigned> GraphBase::getNodeIndex(Operation *op) const {
  for (unsigned idx = 0; idx < nodes.size(); ++idx) {
    if (nodes[idx].op == op)
      return idx;
  }
  return std::nullopt;
}

static bool edgeShapeMatches(const GraphEdge &patternEdge,
                             const GraphEdge &sourceEdge) {
  return patternEdge.producerResultIdx == sourceEdge.producerResultIdx &&
         patternEdge.consumerOperandIdx == sourceEdge.consumerOperandIdx &&
         transformsMatch(patternEdge.transforms, sourceEdge.transforms);
}

/// Verify all edges between the new source node and already-mapped source nodes
/// exactly match the pattern graph. This enforces induced-exact matching.
static bool hasConsistentEdges(const MatchState &state, unsigned patternNodeIdx,
                               unsigned sourceNodeIdx) {
  for (unsigned otherPatternIdx = 0;
       otherPatternIdx < state.pattern.nodes.size(); ++otherPatternIdx) {
    if (!state.mappedPatternNodes[otherPatternIdx] ||
        otherPatternIdx == patternNodeIdx)
      continue;
    unsigned otherSourceIdx = state.patternToSourceNodeIdx[otherPatternIdx];

    for (unsigned edgeIdx : state.pattern.nodes[patternNodeIdx].outEdgeIdxs) {
      const GraphEdge &patternEdge = state.pattern.edges[edgeIdx];
      if (patternEdge.consumerNodeIdx != otherPatternIdx)
        continue;
      const GraphEdge *sourceEdge = state.program.findEdge(
          sourceNodeIdx, otherSourceIdx, patternEdge.producerResultIdx,
          patternEdge.consumerOperandIdx);
      if (!sourceEdge || !edgeShapeMatches(patternEdge, *sourceEdge))
        return false;
    }
    for (unsigned edgeIdx : state.pattern.nodes[patternNodeIdx].inEdgeIdxs) {
      const GraphEdge &patternEdge = state.pattern.edges[edgeIdx];
      if (patternEdge.producerNodeIdx != otherPatternIdx)
        continue;
      const GraphEdge *sourceEdge = state.program.findEdge(
          otherSourceIdx, sourceNodeIdx, patternEdge.producerResultIdx,
          patternEdge.consumerOperandIdx);
      if (!sourceEdge || !edgeShapeMatches(patternEdge, *sourceEdge))
        return false;
    }

    for (unsigned sourceEdgeIdx :
         state.program.nodes[sourceNodeIdx].outEdgeIdxs) {
      const GraphEdge &sourceEdge = state.program.edges[sourceEdgeIdx];
      if (sourceEdge.consumerNodeIdx != otherSourceIdx)
        continue;
      bool matched = false;
      for (unsigned patternEdgeIdx :
           state.pattern.nodes[patternNodeIdx].outEdgeIdxs) {
        const GraphEdge &patternEdge = state.pattern.edges[patternEdgeIdx];
        if (patternEdge.consumerNodeIdx == otherPatternIdx &&
            edgeShapeMatches(patternEdge, sourceEdge)) {
          matched = true;
          break;
        }
      }
      if (!matched)
        return false;
    }
    for (unsigned sourceEdgeIdx :
         state.program.nodes[sourceNodeIdx].inEdgeIdxs) {
      const GraphEdge &sourceEdge = state.program.edges[sourceEdgeIdx];
      if (sourceEdge.producerNodeIdx != otherSourceIdx)
        continue;
      bool matched = false;
      for (unsigned patternEdgeIdx :
           state.pattern.nodes[patternNodeIdx].inEdgeIdxs) {
        const GraphEdge &patternEdge = state.pattern.edges[patternEdgeIdx];
        if (patternEdge.producerNodeIdx == otherPatternIdx &&
            edgeShapeMatches(patternEdge, sourceEdge)) {
          matched = true;
          break;
        }
      }
      if (!matched)
        return false;
    }
  }
  return true;
}

/// Collect the candidate source nodes for the next pattern node using anchor
/// fingerprint matching and already-mapped neighbors as pruning constraints.
static SmallVector<unsigned, 8>
collectCandidateSourceNodes(const MatchState &state, unsigned patternNodeIdx) {
  SmallVector<unsigned, 8> candidates;
  const GraphNode &patternNode = state.pattern.nodes[patternNodeIdx];
  for (unsigned sourceNodeIdx = 0; sourceNodeIdx < state.program.nodes.size();
       ++sourceNodeIdx) {
    if (state.usedSourceNodes[sourceNodeIdx])
      continue;
    const GraphNode &sourceNode = state.program.nodes[sourceNodeIdx];
    if (!patternNode.fp.matches(sourceNode.fp))
      continue;
    if (!hasConsistentEdges(state, patternNodeIdx, sourceNodeIdx))
      continue;
    candidates.push_back(sourceNodeIdx);
  }

  llvm::sort(candidates, [&](unsigned lhs, unsigned rhs) {
    const GraphNode &lhsNode = state.program.nodes[lhs];
    const GraphNode &rhsNode = state.program.nodes[rhs];
    auto lhsKey = std::tuple<unsigned, unsigned, int>(
        lhsNode.inEdgeIdxs.size() + lhsNode.outEdgeIdxs.size(),
        getFingerprintSpecificity(lhsNode.fp), -static_cast<int>(lhs));
    auto rhsKey = std::tuple<unsigned, unsigned, int>(
        rhsNode.inEdgeIdxs.size() + rhsNode.outEdgeIdxs.size(),
        getFingerprintSpecificity(rhsNode.fp), -static_cast<int>(rhs));
    return lhsKey > rhsKey;
  });
  return candidates;
}

static void appendBoundaryInputBindings(SubgraphMatchCandidate &candidate,
                                        const MatchState &state) {
  DenseSet<Operation *> semanticNodeOps;
  for (auto &node : state.program.nodes)
    semanticNodeOps.insert(node.op);

  for (const GraphBoundaryPort &patternPort : state.pattern.boundaryInputs) {
    if (patternPort.boundaryOperandIdx == GraphEdge::kNullIdx)
      continue;

    unsigned sourceNodeIdx =
        candidate.patternToSourceNodeIdx[patternPort.nodeIdx];
    Operation *sourceOp = state.program.nodes[sourceNodeIdx].op;
    auto resolved = resolveValue(sourceOp->getOperand(patternPort.portIdx),
                                 semanticNodeOps);

    SubgraphBoundaryInputBinding binding;
    binding.patternNodeIdx = patternPort.nodeIdx;
    binding.patternPortIdx = patternPort.portIdx;
    binding.boundaryOperandIdx = patternPort.boundaryOperandIdx;
    binding.transforms = std::move(resolved.transforms);
    binding.value = resolved.source;

    if (auto *baseOp = resolved.source.getDefiningOp()) {
      if (auto sourceProducerIdx = state.program.getNodeIndex(baseOp)) {
        if (state.sourceToPatternNodeIdx[*sourceProducerIdx] ==
            GraphEdge::kNullIdx) {
          binding.sourceNodeIdx = *sourceProducerIdx;
          if (auto result = dyn_cast<OpResult>(resolved.source))
            binding.sourcePortIdx = result.getResultNumber();
        }
      }
    }

    candidate.boundaryInputs.push_back(std::move(binding));
  }
}

static LogicalResult
appendBoundaryOutputBindings(SubgraphMatchCandidate &candidate,
                             const MatchState &state) {
  DenseSet<Operation *> semanticNodeOps;
  for (auto &node : state.program.nodes)
    semanticNodeOps.insert(node.op);

  for (const GraphBoundaryPort &patternPort : state.pattern.boundaryOutputs) {
    unsigned sourceNodeIdx =
        candidate.patternToSourceNodeIdx[patternPort.nodeIdx];
    Operation *sourceOp = state.program.nodes[sourceNodeIdx].op;

    auto materialized = resolveUniqueMaterializedOutput(
        sourceOp->getResult(patternPort.portIdx), semanticNodeOps);
    if (failed(materialized))
      return failure();

    SubgraphBoundaryOutputBinding binding;
    binding.patternNodeIdx = patternPort.nodeIdx;
    binding.patternPortIdx = patternPort.portIdx;
    binding.sourceNodeIdx = sourceNodeIdx;
    binding.sourcePortIdx = patternPort.portIdx;
    binding.producedValue = sourceOp->getResult(patternPort.portIdx);
    if (*materialized) {
      binding.materializedValue = (*materialized)->value;
      binding.transforms = std::move((*materialized)->transforms);
    }
    candidate.boundaryOutputs.push_back(std::move(binding));
  }
  return success();
}

static unsigned computeCandidatePriority(const InstructionGraph &pattern) {
  return pattern.nodes.size() * 1000 +
         getFingerprintSpecificity(pattern.nodes[pattern.anchorNodeIdx].fp) *
             100 +
         pattern.edges.size();
}

static LogicalResult
buildMatchCandidate(const MatchState &state, unsigned sourceAnchorNodeIdx,
                    SmallVectorImpl<SubgraphMatchCandidate> &results) {
  SubgraphMatchCandidate candidate;
  candidate.pattern = &state.pattern;
  candidate.patternToSourceNodeIdx = state.patternToSourceNodeIdx;
  candidate.sourceAnchorNodeIdx = sourceAnchorNodeIdx;
  candidate.priority = computeCandidatePriority(state.pattern);

  for (unsigned patternNodeIdx = 0; patternNodeIdx < state.pattern.nodes.size();
       ++patternNodeIdx) {
    candidate.sourceOps.push_back(
        state.program.nodes[state.patternToSourceNodeIdx[patternNodeIdx]].op);
  }

  appendBoundaryInputBindings(candidate, state);
  if (failed(appendBoundaryOutputBindings(candidate, state)))
    return failure();
  results.push_back(std::move(candidate));
  return success();
}

static LogicalResult
matchFromAnchor(MatchState &state, unsigned sourceAnchorNodeIdx,
                SmallVectorImpl<SubgraphMatchCandidate> &results) {
  if (llvm::all_of(state.mappedPatternNodes,
                   [](bool mapped) { return mapped; })) {
    return buildMatchCandidate(state, sourceAnchorNodeIdx, results);
  }

  unsigned nextPatternNodeIdx = chooseNextPatternNode(state);
  auto candidates = collectCandidateSourceNodes(state, nextPatternNodeIdx);

  for (unsigned sourceNodeIdx : candidates) {
    state.patternToSourceNodeIdx[nextPatternNodeIdx] = sourceNodeIdx;
    state.sourceToPatternNodeIdx[sourceNodeIdx] = nextPatternNodeIdx;
    state.mappedPatternNodes[nextPatternNodeIdx] = true;
    state.usedSourceNodes[sourceNodeIdx] = true;
    auto cleanup = llvm::scope_exit([&] {
      state.patternToSourceNodeIdx[nextPatternNodeIdx] = GraphEdge::kNullIdx;
      state.sourceToPatternNodeIdx[sourceNodeIdx] = GraphEdge::kNullIdx;
      state.mappedPatternNodes[nextPatternNodeIdx] = false;
      state.usedSourceNodes[sourceNodeIdx] = false;
    });
    if (failed(matchFromAnchor(state, sourceAnchorNodeIdx, results)))
      return failure();
  }
  return success();
}

static FailureOr<SmallVector<SubgraphMatchCandidate, 4>>
collectMatchesForAnchor(const InstructionGraph &pattern,
                        const ProgramGraph &program,
                        unsigned sourceAnchorNodeIdx) {
  SmallVector<SubgraphMatchCandidate, 4> matches;
  MatchState state{
      pattern,
      program,
      SmallVector<unsigned, 4>(pattern.nodes.size(), GraphEdge::kNullIdx),
      SmallVector<unsigned, 4>(program.nodes.size(), GraphEdge::kNullIdx),
      SmallVector<bool, 4>(pattern.nodes.size(), false),
      SmallVector<bool, 8>(program.nodes.size(), false)};

  state.patternToSourceNodeIdx[pattern.anchorNodeIdx] = sourceAnchorNodeIdx;
  state.sourceToPatternNodeIdx[sourceAnchorNodeIdx] = pattern.anchorNodeIdx;
  state.mappedPatternNodes[pattern.anchorNodeIdx] = true;
  state.usedSourceNodes[sourceAnchorNodeIdx] = true;
  if (failed(matchFromAnchor(state, sourceAnchorNodeIdx, matches)))
    return failure();
  return matches;
}

LogicalResult GraphBase::init(Block &block) {
  DenseMap<Operation *, unsigned> nodeIdxByOp;
  DenseSet<Operation *> semanticNodeOps;
  auto boundaryLayoutOps = collectBoundaryLayoutOps(block);

  for (Operation &op : block.without_terminator()) {
    if (!isSemanticNodeOp(&op, boundaryLayoutOps))
      continue;
    unsigned nodeIdx = nodes.size();
    nodeIdxByOp[&op] = nodeIdx;
    semanticNodeOps.insert(&op);
    nodes.push_back({&op, SemanticFingerprint(&op), {}, {}});
  }
  if (nodes.empty())
    return success();

  for (unsigned nodeIdx = 0; nodeIdx < nodes.size(); ++nodeIdx) {
    Operation *op = nodes[nodeIdx].op;
    for (unsigned operandIdx = 0; operandIdx < op->getNumOperands();
         ++operandIdx) {
      Value operand = op->getOperand(operandIdx);
      if (!isa<RankedTensorType>(operand.getType()))
        continue;

      auto resolved = resolveValue(operand, semanticNodeOps);
      if (resolved.unsupported) {
        return resolved.unsupported->emitError()
               << "dynamic layout transform is not supported by semantic "
                  "matching graph construction";
      }

      if (auto *baseOp = resolved.source.getDefiningOp()) {
        auto it = nodeIdxByOp.find(baseOp);
        if (it != nodeIdxByOp.end()) {
          auto result = dyn_cast<OpResult>(resolved.source);
          if (!result)
            return op->emitError()
                   << "expected internal semantic edge to originate from an op "
                      "result";

          unsigned edgeIdx = edges.size();
          edges.push_back({it->second, nodeIdx, result.getResultNumber(),
                           operandIdx, std::move(resolved.transforms)});
          nodes[it->second].outEdgeIdxs.push_back(edgeIdx);
          nodes[nodeIdx].inEdgeIdxs.push_back(edgeIdx);
          continue;
        }
      }

      unsigned boundaryOperandIdx = GraphEdge::kNullIdx;
      if (auto blockArg = dyn_cast<BlockArgument>(resolved.source))
        boundaryOperandIdx = blockArg.getArgNumber();
      boundaryInputs.push_back({nodeIdx, operandIdx, boundaryOperandIdx,
                                resolved.source,
                                std::move(resolved.transforms)});
    }
  }

  Operation *terminator = block.getTerminator();
  for (Value operand : terminator->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    auto resolved = resolveValue(operand, semanticNodeOps);
    if (resolved.unsupported) {
      return resolved.unsupported->emitError()
             << "dynamic layout transform is not supported by semantic "
                "matching graph construction";
    }
    if (auto result = dyn_cast<OpResult>(resolved.source)) {
      auto it = nodeIdxByOp.find(result.getOwner());
      if (it != nodeIdxByOp.end()) {
        boundaryOutputs.push_back({it->second, result.getResultNumber(),
                                   GraphEdge::kNullIdx, resolved.source,
                                   std::move(resolved.transforms)});
      }
    }
  }

  return success();
}

void GraphBase::dump(raw_ostream &os, StringRef title) const {
  os << title << "\n";
  os << "  nodes: " << nodes.size() << ", edges: " << edges.size()
     << ", boundaryInputs: " << boundaryInputs.size()
     << ", boundaryOutputs: " << boundaryOutputs.size() << "\n";
  for (auto [nodeIdx, node] : llvm::enumerate(nodes)) {
    os << "  node[" << nodeIdx << "] " << node.op->getName() << " fp=";
    printFingerprint(os, node.fp);
    os << " in=" << node.inEdgeIdxs.size() << " out=" << node.outEdgeIdxs.size()
       << "\n";
  }
  for (auto [edgeIdx, edge] : llvm::enumerate(edges)) {
    os << "  edge[" << edgeIdx << "] " << edge.producerNodeIdx << ":"
       << edge.producerResultIdx << " -> " << edge.consumerNodeIdx << ":"
       << edge.consumerOperandIdx << " [";
    printTransformChain(os, edge.transforms);
    os << "]\n";
  }
  for (auto [idx, input] : llvm::enumerate(boundaryInputs)) {
    os << "  boundary-input[" << idx << "] -> " << input.nodeIdx << ":"
       << input.portIdx << " value=";
    printValueRef(os, input.value);
    if (input.boundaryOperandIdx != GraphEdge::kNullIdx)
      os << " boundary-op=" << input.boundaryOperandIdx;
    os << " [";
    printTransformChain(os, input.transforms);
    os << "]\n";
  }
  for (auto [idx, output] : llvm::enumerate(boundaryOutputs)) {
    os << "  boundary-output[" << idx << "] " << output.nodeIdx << ":"
       << output.portIdx << " value=";
    printValueRef(os, output.value);
    os << " [";
    printTransformChain(os, output.transforms);
    os << "]\n";
  }
}

FailureOr<InstructionGraph> InstructionGraph::build(DefineOp defineOp) {
  if (failed(validateComputeRegion(defineOp)))
    return failure();

  InstructionGraph graph;
  graph.instruction = defineOp;
  if (failed(graph.init(defineOp.getSemanticsBlock())))
    return failure();
  if (!graph.nodes.empty()) {
    graph.anchorNodeIdx = selectAnchorNode(graph);
    auto domainNodeIdx = selectDomainNode(defineOp, graph);
    if (failed(domainNodeIdx))
      return failure();
    graph.domainNodeIdx = *domainNodeIdx;
  }
  return std::move(graph);
}

void InstructionGraph::dump() const {
  llvm::dbgs() << "InstructionGraph @" << getDefineName(instruction)
               << " anchor=" << anchorNodeIdx << " domain=" << domainNodeIdx
               << "\n";
  GraphBase::dump(llvm::dbgs(), "  graph");
}

FailureOr<ProgramGraph> ProgramGraph::build(func::FuncOp funcOp) {
  if (funcOp.getBody().getBlocks().size() != 1)
    return funcOp.emitError() << "expected exactly one block in function body";
  ProgramGraph graph;
  graph.funcOp = funcOp;
  if (failed(graph.init(funcOp.getFunctionBody().front())))
    return failure();
  return std::move(graph);
}

void ProgramGraph::dump() {
  llvm::dbgs() << "ProgramGraph @"
               << getFuncName(cast<func::FuncOp>(funcOp.getOperation()))
               << "\n";
  GraphBase::dump(llvm::dbgs(), "  graph");
}

void SubgraphMatchCandidate::dump() const {
  llvm::dbgs() << "MatchCandidate @" << getDefineName(pattern->instruction)
               << " priority=" << priority
               << " sourceAnchor=" << sourceAnchorNodeIdx
               << " patternAnchor=" << pattern->anchorNodeIdx
               << " patternDomain=" << pattern->domainNodeIdx << "\n";
  for (auto [idx, op] : llvm::enumerate(sourceOps))
    llvm::dbgs() << "  sourceOp[" << idx << "] " << op->getName() << "\n";
  for (auto [idx, input] : llvm::enumerate(boundaryInputs)) {
    llvm::dbgs() << "  input[" << idx << "] pattern " << input.patternNodeIdx
                 << ":" << input.patternPortIdx
                 << " boundary-op=" << input.boundaryOperandIdx << " value=";
    printValueRef(llvm::dbgs(), input.value);
    llvm::dbgs() << " [";
    printTransformChain(llvm::dbgs(), input.transforms);
    llvm::dbgs() << "]\n";
  }
  for (auto [idx, output] : llvm::enumerate(boundaryOutputs)) {
    llvm::dbgs() << "  output[" << idx << "] pattern " << output.patternNodeIdx
                 << ":" << output.patternPortIdx << " produced=";
    printValueRef(llvm::dbgs(), output.producedValue);
    if (output.materializedValue) {
      llvm::dbgs() << " materialized=";
      printValueRef(llvm::dbgs(), output.materializedValue);
      llvm::dbgs() << " [";
      printTransformChain(llvm::dbgs(), output.transforms);
      llvm::dbgs() << "]";
    }
    llvm::dbgs() << "\n";
  }
}

void SemanticsGraph::dump() const {
  llvm::dbgs() << "SemanticsGraph @"
               << getFuncName(const_cast<func::FuncOp &>(funcOp)) << "\n";
  llvm::dbgs() << "  nodes=" << nodes.size() << ", edges=" << edges.size()
               << "\n";
  for (auto [nodeIdx, node] : llvm::enumerate(nodes)) {
    llvm::dbgs() << "  node[" << nodeIdx << "] @"
                 << getDefineName(node.instruction)
                 << " anchor=" << node.anchorNodeIdx
                 << " domain=" << node.domainNodeIdx
                 << " sourceOps=" << node.sourceOps.size() << "\n";
    for (auto *op : node.sourceOps)
      llvm::dbgs() << "    " << op->getName() << "\n";
    if (node.domainComputeOp)
      llvm::dbgs() << "    domain-compute=" << node.domainComputeOp->getName()
                   << "\n";
    if (node.domainSourceOp)
      llvm::dbgs() << "    domain-source=" << node.domainSourceOp->getName()
                   << "\n";
    for (auto [idx, input] : llvm::enumerate(node.boundaryInputs)) {
      llvm::dbgs() << "    in[" << idx << "] value=";
      printValueRef(llvm::dbgs(), input.value);
      llvm::dbgs() << " [";
      printTransformChain(llvm::dbgs(), input.transforms);
      llvm::dbgs() << "]\n";
    }
    for (auto [idx, output] : llvm::enumerate(node.boundaryOutputs)) {
      llvm::dbgs() << "    out[" << idx << "] produced=";
      printValueRef(llvm::dbgs(), output.producedValue);
      if (output.materializedValue) {
        llvm::dbgs() << " materialized=";
        printValueRef(llvm::dbgs(), output.materializedValue);
        llvm::dbgs() << " [";
        printTransformChain(llvm::dbgs(), output.transforms);
        llvm::dbgs() << "]";
      }
      llvm::dbgs() << "\n";
    }
  }
  for (auto [edgeIdx, edge] : llvm::enumerate(edges)) {
    llvm::dbgs() << "  edge[" << edgeIdx << "] " << edge.producerNodeIdx << ":"
                 << edge.prodIdx << " -> " << edge.consumerNodeIdx << ":"
                 << edge.consIdx << " [";
    printTransformChain(llvm::dbgs(), edge.transforms);
    llvm::dbgs() << "]\n";
  }
}

FailureOr<InstructionCatalog> InstructionCatalog::build(ModuleOp module) {
  InstructionCatalog catalog;
  for (DefineOp defineOp : module.getOps<DefineOp>()) {
    auto graphOr = InstructionGraph::build(defineOp);
    if (failed(graphOr))
      return failure();
    if (graphOr->nodes.empty())
      continue;
    catalog.graphs.push_back(std::move(*graphOr));
  }

  llvm::sort(catalog.graphs, [](const InstructionGraph &lhs,
                                const InstructionGraph &rhs) {
    return getDefineName(lhs.instruction) < getDefineName(rhs.instruction);
  });

  for (unsigned i = 0; i < catalog.graphs.size(); ++i)
    catalog
        .byAnchor
            [catalog.graphs[i].nodes[catalog.graphs[i].anchorNodeIdx].fp.hash()]
        .push_back(i);

  for (auto &[_, bucket] : catalog.byAnchor) {
    llvm::sort(bucket, [&](unsigned lhsIdx, unsigned rhsIdx) {
      const InstructionGraph &lhs = catalog.graphs[lhsIdx];
      const InstructionGraph &rhs = catalog.graphs[rhsIdx];
      auto lhsKey = std::tuple<unsigned, unsigned, unsigned, StringRef>(
          lhs.nodes.size(),
          getFingerprintSpecificity(lhs.nodes[lhs.anchorNodeIdx].fp),
          lhs.edges.size(), getDefineName(lhs.instruction));
      auto rhsKey = std::tuple<unsigned, unsigned, unsigned, StringRef>(
          rhs.nodes.size(),
          getFingerprintSpecificity(rhs.nodes[rhs.anchorNodeIdx].fp),
          rhs.edges.size(), getDefineName(rhs.instruction));
      return lhsKey > rhsKey;
    });
  }

  return std::move(catalog);
}

SmallVector<const InstructionGraph *, 4>
InstructionCatalog::lookup(const SemanticFingerprint &anchorFp) const {
  SmallVector<const InstructionGraph *, 4> matches;
  auto it = byAnchor.find(anchorFp.hash());
  if (it == byAnchor.end())
    return matches;
  for (unsigned idx : it->second) {
    const InstructionGraph &graph = graphs[idx];
    if (graph.nodes[graph.anchorNodeIdx].fp.matches(anchorFp))
      matches.push_back(&graph);
  }
  return matches;
}

void InstructionCatalog::dump() const {
  llvm::dbgs() << "InstructionCatalog: " << graphs.size() << " graph(s)\n";
  for (const InstructionGraph &graph : graphs)
    graph.dump();
}

static FailureOr<SmallVector<SubgraphMatchCandidate, 8>>
collectCandidatesForFunction(const InstructionCatalog &catalog,
                             const ProgramGraph &program) {
  SmallVector<SubgraphMatchCandidate, 8> candidates;
  for (unsigned sourceNodeIdx = 0; sourceNodeIdx < program.nodes.size();
       ++sourceNodeIdx) {
    const GraphNode &sourceNode = program.nodes[sourceNodeIdx];
    auto patterns = catalog.lookup(sourceNode.fp);
    LLVM_DEBUG({
      llvm::dbgs() << "  source node[" << sourceNodeIdx << "] "
                   << sourceNode.op->getName() << " -> " << patterns.size()
                   << " anchor pattern(s)\n";
    });
    for (const InstructionGraph *pattern : patterns) {
      auto matches = collectMatchesForAnchor(*pattern, program, sourceNodeIdx);
      if (failed(matches))
        return failure();
      for (SubgraphMatchCandidate &match : *matches) {
        LLVM_DEBUG(match.dump());
        candidates.push_back(std::move(match));
      }
    }
  }
  return candidates;
}

static SemanticsGraph
buildSelectedGraph(ProgramGraph &program,
                   SmallVectorImpl<SubgraphMatchCandidate> &candidates) {
  llvm::sort(
      candidates, [](SubgraphMatchCandidate &lhs, SubgraphMatchCandidate &rhs) {
        auto lhsKey = std::tuple<unsigned, unsigned, unsigned, StringRef>(
            lhs.priority, lhs.sourceOps.size(),
            GraphEdge::kNullIdx - lhs.sourceAnchorNodeIdx,
            getDefineName(lhs.getInstruction()));
        auto rhsKey = std::tuple<unsigned, unsigned, unsigned, StringRef>(
            rhs.priority, rhs.sourceOps.size(),
            GraphEdge::kNullIdx - rhs.sourceAnchorNodeIdx,
            getDefineName(rhs.getInstruction()));
        return lhsKey > rhsKey;
      });

  DenseSet<Operation *> coveredOps;
  SmallVector<SubgraphMatchCandidate *, 8> selected;
  for (SubgraphMatchCandidate &candidate : candidates) {
    if (llvm::any_of(candidate.sourceOps,
                     [&](Operation *op) { return coveredOps.contains(op); }))
      continue;
    for (Operation *op : candidate.sourceOps)
      coveredOps.insert(op);
    selected.push_back(&candidate);
  }

  llvm::sort(selected,
             [&](SubgraphMatchCandidate *lhs, SubgraphMatchCandidate *rhs) {
               auto lhsMin = llvm::min_element(lhs->patternToSourceNodeIdx);
               auto rhsMin = llvm::min_element(rhs->patternToSourceNodeIdx);
               return *lhsMin < *rhsMin;
             });

  SemanticsGraph graph;
  graph.funcOp = program.funcOp;

  DenseMap<Value, std::pair<unsigned, unsigned>> producedOutputs;
  for (auto [nodeIdx, candidate] : llvm::enumerate(selected)) {
    SemanticsGraphNode node;
    node.sourceOps = candidate->sourceOps;
    node.anchorNodeIdx = candidate->pattern->anchorNodeIdx;
    node.domainNodeIdx = candidate->pattern->domainNodeIdx;
    node.instruction = candidate->getInstruction();
    node.boundaryOutputs = candidate->boundaryOutputs;
    node.domainComputeOp = candidate->pattern->nodes[node.domainNodeIdx].op;
    unsigned sourceDomainProgramIdx =
        candidate->patternToSourceNodeIdx[node.domainNodeIdx];
    node.domainSourceOp = program.nodes[sourceDomainProgramIdx].op;

    graph.nodes.push_back(std::move(node));
    for (auto [outputIdx, output] :
         llvm::enumerate(candidate->boundaryOutputs)) {
      auto inserted =
          producedOutputs.try_emplace(output.producedValue, nodeIdx, outputIdx);
      assert(inserted.second &&
             "duplicate produced output value in selected graph");
    }
  }

  for (auto [consumerNodeIdx, candidate] : llvm::enumerate(selected)) {
    SemanticsGraphNode &node = graph.nodes[consumerNodeIdx];
    for (auto [inputIdx, input] : llvm::enumerate(candidate->boundaryInputs)) {
      auto producedIt = producedOutputs.find(input.value);
      if (producedIt != producedOutputs.end() &&
          producedIt->second.first != consumerNodeIdx) {
        graph.edges.push_back({producedIt->second.first,
                               static_cast<unsigned>(consumerNodeIdx),
                               producedIt->second.second,
                               input.boundaryOperandIdx, input.transforms});
        continue;
      }
      node.boundaryInputs.push_back(input);
    }
  }

  return graph;
}

FailureOr<SemanticsGraph>
act::runSemanticMatching(func::FuncOp func, InstructionCatalog &catalog) {
  LLVM_DEBUG(llvm::dbgs() << "=== Semantic Matching for function @"
                          << getFuncName(func) << " ===\n");
  auto programOr = ProgramGraph::build(func);
  if (failed(programOr))
    return failure();
  ProgramGraph program = std::move(*programOr);
  if (program.nodes.empty()) {
    func.emitWarning() << "no supported operations found for semantic matching";
    return SemanticsGraph{func, {}, {}};
  }

  LLVM_DEBUG(program.dump());
  auto candidates = collectCandidatesForFunction(catalog, program);
  if (failed(candidates))
    return failure();
  SemanticsGraph graph = buildSelectedGraph(program, *candidates);
  LLVM_DEBUG(graph.dump());
  return graph;
}

FailureOr<SemanticsGraphs> act::runSemanticMatching(ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "=== Semantic Matching (Stage 1) ===\n");

  auto catalog = InstructionCatalog::build(module);
  if (failed(catalog))
    return failure();
  LLVM_DEBUG(catalog->dump());

  SemanticsGraphs graphs;
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    auto programOr = ProgramGraph::build(funcOp);
    if (failed(programOr))
      return failure();
    ProgramGraph program = std::move(*programOr);
    if (program.nodes.empty()) {
      funcOp.emitWarning()
          << "no supported operations found for semantic matching";
      continue;
    }

    LLVM_DEBUG(program.dump());
    auto candidates = collectCandidatesForFunction(*catalog, program);
    if (failed(candidates))
      return failure();
    SemanticsGraph graph = buildSelectedGraph(program, *candidates);
    LLVM_DEBUG(graph.dump());
    graphs.push_back(std::move(graph));
  }

  return graphs;
}
