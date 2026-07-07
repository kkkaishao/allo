#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"

#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h"
#include "allo/Support/AffineValueMapBuilder.h"

namespace mlir::allo {
#define GEN_PASS_DEF_CONVERTLOOPTOSCHEDULEPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::allo;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::loopschedule;

//===----------------------------------------------------------------------===//
// Dependence analysis
//
// Mirrors CIRCT's MemoryDependenceAnalysis but additionally understands Allo
// stream get/put operations (see checkStreamDependence). Both memref and stream
// dependences are recorded into a single MemoryDependenceResult that the
// scheduling problem construction below consumes uniformly.
//===----------------------------------------------------------------------===//

namespace {
struct LoopDependenceAnalysis {
  LoopDependenceAnalysis(func::FuncOp funcOp);

  // Returns the dependences, if any, that the given operation depends on.
  ArrayRef<MemoryDependence> getDependences(Operation *op) {
    return results[op];
  }

  // Redirects the dependences of/to oldOp onto newOp (used when affine
  // structures are lowered to their memref/std equivalents).
  void replaceOp(Operation *oldOp, Operation *newOp);

  MemoryDependenceResult results;
};
} // namespace

static void checkMemrefDependence(SmallVectorImpl<Operation *> &memoryOps,
                                  unsigned depth,
                                  MemoryDependenceResult &results) {
  for (auto *source : memoryOps) {
    for (auto *destination : memoryOps) {
      if (source == destination)
        continue;

      // Initialize the dependence list for this destination.
      if (results.count(destination) == 0)
        results[destination] = SmallVector<MemoryDependence>();

      // Look for inter-iteration dependences on the same memory location.
      affine::MemRefAccess src(source);
      affine::MemRefAccess dst(destination);
      affine::FlatAffineValueConstraints dependenceConstraints;
      SmallVector<affine::DependenceComponent, 2> depComps;

      // Requested depth might not be a valid comparison if they do not belong
      // to the same loop nest
      if (depth > affine::getInnermostCommonLoopDepth({source, destination}))
        continue;

      auto result = affine::checkMemrefAccessDependence(
          src, dst, depth, &dependenceConstraints, &depComps, true);

      results[destination].emplace_back(source, result.value, depComps);

      // Also consider intra-iteration dependences on the same memory location.
      // This currently does not consider aliasing.
      if (src != dst)
        continue;

      // Collect surrounding loops to use in dependence components. Only proceed
      // if we are in the innermost loop.
      SmallVector<affine::AffineForOp> enclosingLoops;
      affine::getAffineForIVs(*destination, &enclosingLoops);
      if (enclosingLoops.size() != depth)
        continue;

      // Look for the common parent that src and dst share. If there is none,
      // there is nothing more to do.
      SmallVector<Operation *> srcParents;
      affine::getEnclosingAffineOps(*source, &srcParents);
      SmallVector<Operation *> dstParents;
      affine::getEnclosingAffineOps(*destination, &dstParents);

      Operation *commonParent = nullptr;
      for (auto *srcParent : llvm::reverse(srcParents)) {
        for (auto *dstParent : llvm::reverse(dstParents)) {
          if (srcParent == dstParent)
            commonParent = srcParent;
          if (commonParent != nullptr)
            break;
        }
        if (commonParent != nullptr)
          break;
      }

      if (commonParent == nullptr)
        continue;

      // Check the common parent's regions.
      for (auto &commonRegion : commonParent->getRegions()) {
        if (commonRegion.empty())
          continue;

        // Only support structured constructs with single-block regions for now.
        assert(commonRegion.hasOneBlock() &&
               "only single-block regions are supported");

        Block &commonBlock = commonRegion.front();

        // Find the src and dst ancestor in the common block, if any.
        Operation *srcOrAncestor = commonBlock.findAncestorOpInBlock(*source);
        Operation *dstOrAncestor =
            commonBlock.findAncestorOpInBlock(*destination);
        if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
          continue;

        // Check if the src or its ancestor is before the dst or its ancestor.
        if (srcOrAncestor->isBeforeInBlock(dstOrAncestor)) {
          // Build dependence components for each loop depth.
          SmallVector<affine::DependenceComponent> intraDeps;
          for (size_t i = 0; i < depth; ++i) {
            affine::DependenceComponent depComp;
            depComp.op = enclosingLoops[i];
            depComp.lb = 0;
            depComp.ub = 0;
            intraDeps.push_back(depComp);
          }

          results[destination].emplace_back(
              source, affine::DependenceResult::HasDependence, intraDeps);
        }
      }
    }
  }
}

// Returns the base stream SSA value a stream get/put operates on. Two accesses
// on different bases are always independent (SSA identity is a precise
// disambiguation for streams, which are not reassigned through aliases).
static Value getStreamBase(Operation *op) {
  if (auto get = dyn_cast<StreamGetOp>(op))
    return get.getStream();
  return cast<StreamPutOp>(op).getStream();
}

// Returns the FIFO-selecting indices of a stream get/put operation.
static OperandRange getStreamIndices(Operation *op) {
  if (auto get = dyn_cast<StreamGetOp>(op))
    return get.getIndices();
  return cast<StreamPutOp>(op).getIndices();
}

// Nearest enclosing affine.for, skipping non-loop parents (e.g. affine.if).
static affine::AffineForOp getNearestAffineFor(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto forOp = dyn_cast<affine::AffineForOp>(parent))
      return forOp;
  return nullptr;
}

// Whether two same-base stream accesses may touch the same FIFO. A stream value
// is an array of FIFOs selected by its indices, so this is an affine
// disambiguation on the indices, analogous to array-subscript aliasing.
namespace {
enum class FifoAlias { Same, Distinct, Unknown };
} // namespace

static FifoAlias compareFifo(AffineValueMapBuilder &builder, Operation *a,
                             Operation *b) {
  builder.reset();
  for (Value idx : getStreamIndices(a))
    if (failed(builder.importValue(idx)))
      return FifoAlias::Unknown;
  affine::AffineValueMap ma = builder.compose();

  builder.reset();
  for (Value idx : getStreamIndices(b))
    if (failed(builder.importValue(idx)))
      return FifoAlias::Unknown;
  affine::AffineValueMap mb = builder.compose();

  if (ma.getNumResults() != mb.getNumResults())
    return FifoAlias::Unknown;

  affine::AffineValueMap diff;
  affine::AffineValueMap::difference(ma, mb, &diff);
  bool allZero = true;
  for (AffineExpr e : diff.getAffineMap().getResults()) {
    auto cst = dyn_cast<AffineConstantExpr>(e);
    if (!cst) {
      // Symbolic offset: cannot prove same or distinct FIFO.
      allZero = false;
      continue;
    }
    if (cst.getValue() != 0)
      return FifoAlias::Distinct; // some coordinate differs by a constant
  }
  return allZero ? FifoAlias::Same : FifoAlias::Unknown;
}

// Build dependence components mirroring the op's enclosing loop nest, placing
// `distance` on the innermost loop (the only component the scheduler reads).
static SmallVector<affine::DependenceComponent>
streamDepComponents(Operation *op, int64_t distance) {
  SmallVector<affine::AffineForOp> loops;
  affine::getAffineForIVs(*op, &loops);
  SmallVector<affine::DependenceComponent> comps;
  for (auto loop : loops) {
    affine::DependenceComponent comp;
    comp.op = loop;
    comp.lb = 0;
    comp.ub = 0;
    comps.push_back(comp);
  }
  assert(!comps.empty() && "stream op must be enclosed by a loop");
  comps.back().lb = distance;
  return comps;
}

// Streams are FIFOs: every pair of accesses to the same FIFO must preserve its
// program+iteration order, regardless of direction (unlike memory, get-get is
// ordered and there is no RAW/WAR/WAW distinction). Each may-aliasing pair is
// serialized with a distance-0 intra-iteration edge plus a distance-1
// loop-carried back edge, closing the recurrence that bounds the II. Results
// are written into the same map the memref analysis populates so downstream
// scheduling consumes them uniformly.
static void checkStreamDependence(SmallVectorImpl<Operation *> &streamOps,
                                  AffineValueMapBuilder &builder,
                                  MemoryDependenceResult &results) {
  for (unsigned i = 0, e = streamOps.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      // `earlier` precedes `later` in program order: `walk` is a pre-order
      // traversal, so a smaller index is never scheduled after a larger one.
      Operation *earlier = streamOps[i];
      Operation *later = streamOps[j];

      if (getStreamBase(earlier) != getStreamBase(later))
        continue;

      // Only serialize accesses sharing the same innermost loop, so both ends
      // of the edge land in a single scheduling problem.
      affine::AffineForOp loop = getNearestAffineFor(earlier);
      if (!loop || loop != getNearestAffineFor(later))
        continue;

      // Provably-distinct FIFOs are independent; same or unknown are ordered.
      if (compareFifo(builder, earlier, later) == FifoAlias::Distinct)
        continue;

      results[later].emplace_back(earlier,
                                  affine::DependenceResult::HasDependence,
                                  streamDepComponents(later, /*distance=*/0));
      results[earlier].emplace_back(
          later, affine::DependenceResult::HasDependence,
          streamDepComponents(earlier, /*distance=*/1));
    }
  }
}

LoopDependenceAnalysis::LoopDependenceAnalysis(func::FuncOp funcOp) {
  std::vector<SmallVector<affine::AffineForOp, 2>> depthToLoops;
  affine::gatherLoops(funcOp, depthToLoops);

  SmallVector<Operation *> memoryOps;
  SmallVector<Operation *> streamOps;
  funcOp->walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      memoryOps.push_back(op);
    } else if (isa<StreamGetOp, StreamPutOp>(op)) {
      streamOps.push_back(op);
    }
  });

  for (unsigned d = 0; d < depthToLoops.size(); ++d)
    checkMemrefDependence(memoryOps, d, results);

  AffineValueMapBuilder builder(funcOp.getContext());
  checkStreamDependence(streamOps, builder, results);
}

void LoopDependenceAnalysis::replaceOp(Operation *oldOp, Operation *newOp) {
  // Move the dependence list keyed on oldOp over to newOp.
  auto it = results.find(oldOp);
  if (it != results.end()) {
    results[newOp] = std::move(it->second);
    results.erase(it);
  }

  // Redirect any dependences that originate from oldOp.
  for (auto &entry : results)
    for (auto &dep : entry.second)
      if (dep.source == oldOp)
        dep.source = newOp;
}

//===----------------------------------------------------------------------===//
// Affine structure lowering
//
// Materialize affine loads/stores/ifs with memref/std/scf ops so the schedule
// sees explicit address and condition computations. Copied from CIRCT's
// AffineToLoopSchedule, updating the dependence analysis alongside each
// rewrite.
//===----------------------------------------------------------------------===//

namespace {
class AffineLoadLowering : public OpConversionPattern<AffineLoadOp> {
public:
  AffineLoadLowering(MLIRContext *context, LoopDependenceAnalysis &dependences)
      : OpConversionPattern(context), dependences(dependences) {}

  LogicalResult
  matchAndRewrite(AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    auto memrefLoad = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, op.getMemRef(), *resultOperands);
    dependences.replaceOp(op, memrefLoad);
    return success();
  }

private:
  LoopDependenceAnalysis &dependences;
};

class AffineStoreLowering : public OpConversionPattern<AffineStoreOp> {
public:
  AffineStoreLowering(MLIRContext *context, LoopDependenceAnalysis &dependences)
      : OpConversionPattern(context), dependences(dependences) {}

  LogicalResult
  matchAndRewrite(AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto memrefStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    dependences.replaceOp(op, memrefStore);
    return success();
  }

private:
  LoopDependenceAnalysis &dependences;
};

// Hoist computation out of scf::IfOp branches, turning it into a mux-like
// operation and exposing potentially concurrent execution of its branches.
struct IfOpHoisting : OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&]() {
      if (!op.thenBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.thenBlock(), --op.thenBlock()->end());
        rewriter.inlineBlockBefore(&op.getThenRegion().front(), op);
      }
      if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.elseBlock(), --op.elseBlock()->end());
        rewriter.inlineBlockBefore(&op.getElseRegion().front(), op);
      }
    });
    return success();
  }
};
} // namespace

static bool ifOpLegalityCallback(IfOp op) {
  return op.thenBlock()->without_terminator().empty() &&
         (!op.elseBlock() || op.elseBlock()->without_terminator().empty());
}

static bool yieldOpLegalityCallback(AffineYieldOp op) {
  return !op->getParentOfType<IfOp>();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertLoopToSchedulePass
    : allo::impl::ConvertLoopToSchedulePassBase<ConvertLoopToSchedulePass> {
  void runOnOperation() override;

private:
  LogicalResult lowerAffineStructures(LoopDependenceAnalysis &dependences);
  ModuloProblem buildModuloProblem(AffineForOp forOp,
                                   LoopDependenceAnalysis &deps);
  LogicalResult populateOperatorTypes(AffineForOp forOp,
                                      ModuloProblem &problem);
  LogicalResult solveSchedulingProblem(AffineForOp forOp,
                                       ModuloProblem &problem);
  LogicalResult
  createLoopSchedulePipeline(SmallVectorImpl<AffineForOp> &loopNest,
                             ModuloProblem &problem);
};
} // namespace

LogicalResult ConvertLoopToSchedulePass::lowerAffineStructures(
    LoopDependenceAnalysis &dependences) {
  auto *context = &getContext();
  auto op = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, ArithDialect, MemRefDialect,
                         SCFDialect>();
  target.addIllegalOp<AffineIfOp, AffineLoadOp, AffineStoreOp>();
  target.addDynamicallyLegalOp<IfOp>(ifOpLegalityCallback);
  target.addDynamicallyLegalOp<AffineYieldOp>(yieldOpLegalityCallback);

  RewritePatternSet patterns(context);
  populateAffineToStdConversionPatterns(patterns);
  patterns.add<AffineLoadLowering>(context, dependences);
  patterns.add<AffineStoreLowering>(context, dependences);
  patterns.add<IfOpHoisting>(context);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  return success();
}

// Build a modulo scheduling problem for the given (innermost) loop, seeded with
// the memory/stream dependences plus the structural dependences (conditionals,
// side-effect anchor, and loop-carried iter args). Mirrors CIRCT's
// CyclicSchedulingAnalysis::analyzeForOp, extended to anchor stream accesses.
ModuloProblem
ConvertLoopToSchedulePass::buildModuloProblem(AffineForOp forOp,
                                              LoopDependenceAnalysis &deps) {
  ModuloProblem problem(forOp);

  // Insert memory and stream dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    problem.insertOperation(op);

    for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence.
      unsigned distance = *memoryDep.dependenceComponents.back().lb;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Insert conditional dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    Block *thenBlock = nullptr;
    Block *elseBlock = nullptr;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      thenBlock = ifOp.thenBlock();
      elseBlock = ifOp.elseBlock();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      thenBlock = ifOp.getThenBlock();
      if (ifOp.hasElse())
        elseBlock = ifOp.getElseBlock();
    } else {
      return WalkResult::advance();
    }

    // No special handling required for control-only `if`s.
    if (op->getNumResults() == 0)
      return WalkResult::skip();

    // Model the implicit value flow from the `yield` to the `if`'s result(s).
    Problem::Dependence depThen(thenBlock->getTerminator(), op);
    auto depInserted = problem.insertDependence(depThen);
    assert(succeeded(depInserted));
    (void)depInserted;

    if (elseBlock) {
      Problem::Dependence depElse(elseBlock->getTerminator(), op);
      depInserted = problem.insertDependence(depElse);
      assert(succeeded(depInserted));
      (void)depInserted;
    }

    return WalkResult::advance();
  });

  // Anchor: side-effecting ops (stores and stream accesses) must be scheduled
  // before the loop terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  forOp.getBody()->walk([&](Operation *op) {
    if (!isa<AffineStoreOp, memref::StoreOp, StreamGetOp, StreamPutOp>(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  // Handle explicitly computed loop-carried values, i.e. excluding the
  // induction variable. Insert inter-iteration dependences from the definers of
  // "iter_args" to their users.
  if (unsigned nIterArgs = anchor->getNumOperands(); nIterArgs > 0) {
    auto iterArgs = forOp.getRegionIterArgs();
    for (unsigned i = 0; i < nIterArgs; ++i) {
      Operation *iterArgDefiner = anchor->getOperand(i).getDefiningOp();
      // If it's not an operation, we don't need to model the dependence.
      if (!iterArgDefiner)
        continue;

      for (Operation *iterArgUser : iterArgs[i].getUsers()) {
        Problem::Dependence dep(iterArgDefiner, iterArgUser);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;

        // Values always flow between subsequent iterations.
        problem.setDistance(dep, 1);
      }
    }
  }

  return problem;
}

// Populate the scheduling problem operator types. We assume Calyx-like operator
// latencies, extended with a per-stream operator/resource for stream accesses.
LogicalResult
ConvertLoopToSchedulePass::populateOperatorTypes(AffineForOp forOp,
                                                 ModuloProblem &problem) {
  // A minimal operator library; ultimately this should come from a dialect
  // interface in the Scheduling dialect.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);

  // Assign a limited operator+resource keyed on a memory or stream handle.
  auto setLimitedResource = [&](Operation *op, Value handle, StringRef prefix) {
    auto key = (prefix + std::to_string(hash_value(handle))).str();
    Problem::OperatorType opr = problem.getOrInsertOperatorType(key);
    problem.setLatency(opr, 1);
    problem.setLinkedOperatorType(op, opr);

    auto rsrc = problem.getOrInsertResourceType(key + "_rsrc");
    problem.setLimit(rsrc, 1);
    problem.setLinkedResourceTypes(op,
                                   SmallVector<Problem::ResourceType>{rsrc});
  };

  Operation *unsupported = nullptr;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AddIOp, SubIOp, IfOp, AffineYieldOp, arith::ConstantOp, CmpIOp,
              IndexCastOp, ExtSIOp, ExtUIOp, TruncIOp, memref::AllocaOp,
              scf::YieldOp>([&](Operation *combOp) {
          // Some known combinational ops.
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<AffineStoreOp, memref::StoreOp>([&](Operation *memOp) {
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<memref::StoreOp>(*memOp).getMemRef();
          setLimitedResource(memOp, memRef, "mem_");
          return WalkResult::advance();
        })
        .Case<AffineLoadOp, memref::LoadOp>([&](Operation *memOp) {
          Value memRef = isa<AffineLoadOp>(*memOp)
                             ? cast<AffineLoadOp>(*memOp).getMemRef()
                             : cast<memref::LoadOp>(*memOp).getMemRef();
          setLimitedResource(memOp, memRef, "mem_");
          return WalkResult::advance();
        })
        .Case<StreamGetOp, StreamPutOp>([&](Operation *streamOp) {
          // A stream access takes one cycle. Same-FIFO accesses are already
          // serialized by the dependence recurrence built in the analysis, so
          // no shared port resource is modeled here (which would wrongly
          // serialize accesses to distinct FIFOs of the same stream array).
          Problem::OperatorType streamOpr =
              problem.getOrInsertOperatorType("stream");
          problem.setLatency(streamOpr, 1);
          problem.setLinkedOperatorType(streamOp, streamOpr);
          return WalkResult::advance();
        })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Some known multi-cycle ops.
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return forOp.emitError("unsupported operation ") << *unsupported;

  return success();
}

LogicalResult
ConvertLoopToSchedulePass::solveSchedulingProblem(AffineForOp forOp,
                                                  ModuloProblem &problem) {
  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = forOp.getBody()->getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  return success();
}

/// Create the loopschedule pipeline op for a loop nest. Copied from CIRCT's
/// AffineToLoopSchedule; the generic clone handles stream operations.
LogicalResult ConvertLoopToSchedulePass::createLoopSchedulePipeline(
    SmallVectorImpl<AffineForOp> &loopNest, ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loopNest.back();

  auto outerLoop = loopNest.front();
  auto innerLoop = loopNest.back();
  ImplicitLocOpBuilder builder(outerLoop.getLoc(), outerLoop);

  // Create Values for the loop's lower and upper bounds.
  Value lowerBound = lowerAffineLowerBound(innerLoop, builder);
  Value upperBound = lowerAffineUpperBound(innerLoop, builder);
  int64_t stepValue = innerLoop.getStep().getSExtValue();
  auto step = arith::ConstantOp::create(
      builder, IntegerAttr::get(builder.getIndexType(), stepValue));

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = innerLoop.getResultTypes();

  auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  SmallVector<Value> iterArgs;
  iterArgs.push_back(lowerBound);
  iterArgs.append(innerLoop.getInits().begin(), innerLoop.getInits().end());

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount = getConstantTripCount(forOp))
    tripCountAttr = builder.getI64IntegerAttr(*tripCount);

  auto pipeline = LoopSchedulePipelineOp::create(builder, resultTypes, ii,
                                                 tripCountAttr, iterArgs);

  // Create the condition, which currently just compares the induction variable
  // to the upper bound.
  Block &condBlock = pipeline.getCondBlock();
  builder.setInsertionPointToStart(&condBlock);
  auto cmpResult = arith::CmpIOp::create(builder, builder.getI1Type(),
                                         arith::CmpIPredicate::ult,
                                         condBlock.getArgument(0), upperBound);
  condBlock.getTerminator()->insertOperands(0, {cmpResult});

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, scf::YieldOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  IRMapping valueMap;
  // Nested loops are not supported yet.
  assert(iterArgs.size() == forOp.getBody()->getNumArguments());
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(forOp.getBody()->getArgument(i),
                 pipeline.getStagesBlock().getArgument(i));

  // Create the stages.
  Block &stagesBlock = pipeline.getStagesBlock();
  builder.setInsertionPointToStart(&stagesBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DominanceInfo dom(getOperation());

  // Keys for translating values in each stage
  SmallVector<SmallVector<Value>> registerValues;
  SmallVector<SmallVector<Type>> registerTypes;

  // The maps that ensure a stage uses the correct version of a value
  SmallVector<IRMapping> stageValueMaps;

  // For storing the range of stages an operation's results need to be valid for
  DenseMap<Operation *, std::pair<unsigned, unsigned>> pipeTimes;

  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];

    // Collect the return types for this stage. Operations whose results are not
    // used within this stage are returned.
    auto isLoopTerminator = [forOp](Operation *op) {
      return isa<AffineYieldOp>(op) && op->getParentOp() == forOp;
    };

    // Initialize set of registers up until this point in time
    for (unsigned i = registerValues.size(); i <= startTime; ++i)
      registerValues.emplace_back(SmallVector<Value>());

    // Check each operation to see if its results need plumbing
    for (auto *op : group) {
      if (op->getUsers().empty())
        continue;

      unsigned pipeEndTime = 0;
      for (auto *user : op->getUsers()) {
        unsigned userStartTime = *problem.getStartTime(user);
        if (*problem.getStartTime(user) > startTime)
          pipeEndTime = std::max(pipeEndTime, userStartTime);
        else if (isLoopTerminator(user))
          // Manually forward the value into the terminator's valueMap
          pipeEndTime = std::max(pipeEndTime, userStartTime + 1);
      }

      // Insert the range of pipeline stages the value needs to be valid for
      pipeTimes[op] = std::pair(startTime, pipeEndTime);

      // Add register stages for each time slice we need to pipe to
      for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
        registerValues.push_back(SmallVector<Value>());

      // Keep a collection of this stages results as keys to our valueMaps
      for (auto result : op->getResults())
        registerValues[startTime].push_back(result);

      // Other stages that use the value will need these values as keys too
      unsigned firstUse = std::max(
          startTime + 1,
          startTime + *problem.getLatency(*problem.getLinkedOperatorType(op)));
      for (unsigned i = firstUse; i < pipeEndTime; ++i) {
        for (auto result : op->getResults())
          registerValues[i].push_back(result);
      }
    }
  }

  // Now make register Types and stageValueMaps
  for (unsigned i = 0; i < registerValues.size(); ++i) {
    SmallVector<mlir::Type> types;
    for (auto val : registerValues[i])
      types.push_back(val.getType());

    registerTypes.push_back(types);
    stageValueMaps.push_back(valueMap);
  }

  // One more map is needed for the pipeline stages terminator
  stageValueMaps.push_back(valueMap);

  // Create stages along with maps
  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];
    llvm::sort(group, [&](Operation *a, Operation *b) {
      return dom.properlyDominates(a, b);
    });
    auto stageTypes = registerTypes[startTime];
    // Add the induction variable increment in the first stage.
    if (startTime == 0)
      stageTypes.push_back(lowerBound.getType());

    // Create the stage itself.
    builder.setInsertionPoint(stagesBlock.getTerminator());
    auto startTimeAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), startTime);
    auto stage =
        LoopSchedulePipelineStageOp::create(builder, stageTypes, startTimeAttr);
    auto &stageBlock = stage.getBodyBlock();
    auto *stageTerminator = stageBlock.getTerminator();
    builder.setInsertionPointToStart(&stageBlock);

    for (auto *op : group) {
      auto *newOp = builder.clone(*op, stageValueMaps[startTime]);

      // All further uses in this stage should used the cloned-version of values
      // So we update the mapping in this stage
      for (auto result : op->getResults())
        stageValueMaps[startTime].map(
            result, newOp->getResult(result.getResultNumber()));
    }

    // Register all values in the terminator, using their mapped value
    SmallVector<Value> stageOperands;
    unsigned resIndex = 0;
    for (auto res : registerValues[startTime]) {
      stageOperands.push_back(stageValueMaps[startTime].lookup(res));
      // Additionally, update the map of the stage that will consume the
      // registered value
      unsigned destTime = startTime + 1;
      unsigned latency = *problem.getLatency(
          *problem.getLinkedOperatorType(res.getDefiningOp()));
      // Multi-cycle case
      if (*problem.getStartTime(res.getDefiningOp()) == startTime &&
          latency > 1)
        destTime = startTime + latency;
      destTime = std::min((unsigned)(stageValueMaps.size() - 1), destTime);
      stageValueMaps[destTime].map(res, stage.getResult(resIndex++));
    }
    // Add these mapped values to pipeline.register
    stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                    stageOperands);

    // Add the induction variable increment to the first stage.
    if (startTime == 0) {
      auto incResult =
          arith::AddIOp::create(builder, stagesBlock.getArgument(0), step);
      stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                      incResult->getResults());
    }
  }

  // Add the iter args and results to the terminator.
  auto stagesTerminator =
      cast<LoopScheduleTerminatorOp>(stagesBlock.getTerminator());

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;
  termIterArgs.push_back(
      stagesBlock.front().getResult(stagesBlock.front().getNumResults() - 1));

  for (auto value : forOp.getBody()->getTerminator()->getOperands()) {
    unsigned lookupTime = std::min((unsigned)(stageValueMaps.size() - 1),
                                   pipeTimes[value.getDefiningOp()].second);

    termIterArgs.push_back(stageValueMaps[lookupTime].lookup(value));
    termResults.push_back(stageValueMaps[lookupTime].lookup(value));
  }

  stagesTerminator.getIterArgsMutable().append(termIterArgs);
  stagesTerminator.getResultsMutable().append(termResults);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(pipeline.getResult(i));

  // Remove the loop nest from the IR.
  loopNest.front().walk([](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

void ConvertLoopToSchedulePass::runOnOperation() {
  auto funcOp = getOperation();
  if (funcOp.getFunctionBody().empty())
    return;

  // Dependence analysis (memory + stream), kept in sync through affine
  // lowering below.
  auto &dependences = getAnalysis<LoopDependenceAnalysis>();

  // Materialize affine structures so the schedule sees explicit addresses.
  if (failed(lowerAffineStructures(dependences)))
    return signalPassFailure();

  // Collect and schedule the top-level loops. Restrict to single (non-nested)
  // loops to keep things simple for now, matching CIRCT.
  Block &entry = funcOp.getFunctionBody().front();
  for (auto root : llvm::make_early_inc_range(entry.getOps<AffineForOp>())) {
    SmallVector<AffineForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, root);
    if (nestedLoops.size() != 1)
      continue;

    AffineForOp loop = nestedLoops.back();
    ModuloProblem problem = buildModuloProblem(loop, dependences);

    if (failed(populateOperatorTypes(loop, problem)))
      return signalPassFailure();
    if (failed(solveSchedulingProblem(loop, problem)))
      return signalPassFailure();
    if (failed(createLoopSchedulePipeline(nestedLoops, problem)))
      return signalPassFailure();
  }
}
