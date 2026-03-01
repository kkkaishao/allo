#include "allo/Analysis/DataflowAnalysis.h"
#include "allo/TransformOps/AlloTransformOps.h"
#include "allo/TransformOps/Utils.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include <optional>

using namespace mlir;
using namespace mlir::allo;

///===----------------------------------------------------------------------===//
/// MapGridOp
///===----------------------------------------------------------------------===//
namespace {
using Pid = SmallVector<int32_t, 4>;
using PidList = SmallVector<Pid, 8>;
} // namespace

static void enumeratePids(ArrayRef<int32_t> mapping, unsigned dim, Pid &current,
                          PidList &out) {
  if (dim == mapping.size()) {
    out.push_back(current);
    return;
  }
  for (auto v = 0; v < mapping[dim]; ++v) {
    current.push_back(v);
    enumeratePids(mapping, dim + 1, current, out);
    current.pop_back();
  }
}

static std::string buildInstanceName(StringRef base, ArrayRef<int32_t> pid) {
  std::string name = base.str() + "__vm";
  for (int32_t v : pid)
    name += "_" + std::to_string(v);
  return name;
}

DiagnosedSilenceableFailure
transform::MapGridOp::applyToOne(transform::TransformRewriter &rewriter,
                                 Operation *target,
                                 transform::ApplyToEachResultList &results,
                                 transform::TransformState &state) {
  auto k = dyn_cast<allo::KernelOp>(target);
  if (!k) {
    return emitSilenceableFailure(target)
           << "expected allo.kernel op, got " << target->getName();
  }
  auto givenGrid = getGrid();
  // if given, use the mapping from the op attribute
  auto grid = givenGrid.has_value() ? *givenGrid : k.getGrid();
  // if no mapping is needed, skip the transformation
  if (isTrivialMapping(grid))
    return DiagnosedSilenceableFailure::success();

  // TODO: support non-void kernels
  if (k.getFunctionType().getNumResults() != 0) {
    return emitSilenceableFailure(target)
           << "apply-virtual-map only supports void allo.kernel: @"
           << k.getName();
  }

  // get all pid combinations
  PidList pids;
  {
    Pid current;
    enumeratePids(grid, /*dim=*/0, current, pids);
  }

  // prepare for constant propagation after cloning the kernel
  PassManager pm(rewriter.getContext());
  if (getSccp()) {
    // run SCCP for aggressive constant propagation
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createSCCPPass());
  }
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  auto mod = k->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(mod);
  SmallVector<std::string, 8> instNames;

  for (auto pid : pids) {
    auto kName = buildInstanceName(k.getSymName(), pid);
    if (symbolTable.lookup(kName)) {
      return emitSilenceableFailure(target)
             << "materialize-virtual-mapping instance symbol collision: @"
             << kName;
    }
    instNames.push_back(kName);
    // clone the kernel
    rewriter.setInsertionPoint(k);
    auto clone = cast<allo::KernelOp>(rewriter.clone(*k));
    clone->setAttr(SymbolTable::getSymbolAttrName(),
                   rewriter.getStringAttr(kName));
    clone->setAttr(clone.getGridAttrName(), rewriter.getDenseI32ArrayAttr(1));

    // replace get_pid and get_n_progs to constants
    SmallVector<Operation *, 8> opToErase;
    clone.walk([&](Operation *op) {
      if (auto pidOp = dyn_cast<GetProgramIdOp>(op)) {
        int64_t dim = pidOp.getAxiAttr().getInt();
        rewriter.setInsertionPoint(pidOp);
        auto cst =
            arith::ConstantIndexOp::create(rewriter, pidOp.getLoc(), pid[dim]);
        rewriter.replaceAllOpUsesWith(pidOp, cst);
        opToErase.push_back(pidOp);
      } else if (auto nProgsOp = dyn_cast<GetNumProgramsOp>(op)) {
        int64_t dim = nProgsOp.getAxi();
        rewriter.setInsertionPoint(nProgsOp);
        auto cst = arith::ConstantIndexOp::create(rewriter, nProgsOp.getLoc(),
                                                  grid[dim]);
        rewriter.replaceAllOpUsesWith(nProgsOp, cst);
        opToErase.push_back(nProgsOp);
      }
    });
    for (auto *op : opToErase)
      rewriter.eraseOp(op);

    // constant propagation
    if (failed(pm.run(clone))) {
      return emitSilenceableFailure(target)
             << "failed to perform constant propagation for the cloned kernel "
                "instance";
    }
  }
  // rewrite calls
  auto mayCalls = SymbolTable::getSymbolUses(k.getSymNameAttr(), mod);
  if (!mayCalls) {
    target->erase();
    return DiagnosedSilenceableFailure::success();
  }
  for (auto use : *mayCalls) {
    auto call = dyn_cast<CallOpInterface>(use.getUser());
    assert(call && "expected call op");
    rewriter.setInsertionPoint(call);
    for (const auto &inst : instNames) {
      allo::CallOp::create(rewriter, call.getLoc(), inst,
                           call->getResultTypes(), call.getArgOperands());
    }
    rewriter.eraseOp(call);
  }
  // erase the original kernel
  k.erase();
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::MapGridOp::verify() {
  auto grid = getGrid();
  if (grid.has_value()) {
    if (llvm::any_of(*grid, [](int32_t v) { return v <= 0; })) {
      return emitOpError() << "invalid grid mapping: " << *grid;
    }
  }
  return success();
}

///===----------------------------------------------------------------------===//
/// ChainOp
///===----------------------------------------------------------------------===//
static std::string makeMergedKernelName(StringRef first, StringRef second) {
  return first.str() + "-" + second.str();
}

static void mergeKernelAttrs(KernelOp first, KernelOp second,
                             NamedAttrList &kernelAttr,
                             SmallVectorImpl<DictionaryAttr> &argAttrs) {
  // merge kernel attributes
  for (auto attr : first->getDiscardableAttrDictionary()) {
    kernelAttr.set(attr.getName(), attr.getValue());
  }
  for (auto attr : second->getDiscardableAttrDictionary()) {
    assert(!kernelAttr.getNamed(attr.getName()).has_value() &&
           "conflicting kernel attributes when merging kernels");
    kernelAttr.set(attr.getName(), attr.getValue());
  }
  // merge arg attributes
  if (auto firstArgAttrs = first.getArgAttrsAttr()) {
    llvm::append_range(argAttrs, firstArgAttrs.getAsRange<DictionaryAttr>());
  } else {
    argAttrs.append(first.getNumArguments(),
                    DictionaryAttr::get(first.getContext()));
  }
  if (auto secondArgAttrs = second.getArgAttrsAttr()) {
    llvm::append_range(argAttrs, secondArgAttrs.getAsRange<DictionaryAttr>());
  } else {
    argAttrs.append(second.getNumArguments(),
                    DictionaryAttr::get(second.getContext()));
  }
}

static allo::KernelOp createMergedKernel(RewriterBase &b, KernelOp first,
                                         KernelOp second, IRMapping &mapper) {
  OpBuilder::InsertionGuard g(b);
  // merge arguments
  auto argTys = llvm::to_vector(first.getArgumentTypes());
  llvm::append_range(argTys, second.getArgumentTypes());
  if (!first.getResultTypes().empty() || !second.getResultTypes().empty()) {
    // TODO: support non-void kernels
    return nullptr;
  }
  auto funcTy = FunctionType::get(b.getContext(), argTys, {});
  b.setInsertionPoint(first);
  auto funcName = makeMergedKernelName(first.getSymName(), second.getSymName());
  // merge attributes
  NamedAttrList attrList(first->getDiscardableAttrDictionary());
  SmallVector<DictionaryAttr, 8> argAttrs;
  mergeKernelAttrs(first, second, attrList, argAttrs);
  // create merged kernel
  auto merged = allo::KernelOp::create(b, first.getLoc(), funcName, funcTy,
                                       attrList, argAttrs, 1);
  Block *body = merged.addEntryBlock();
  b.setInsertionPointToEnd(body);
  // map from old args to new args
  unsigned nFirstArgs = first.getNumArguments();
  unsigned nSecondArgs = second.getNumArguments();
  for (unsigned i = 0; i < nFirstArgs; ++i) {
    mapper.map(first.getArgument(i), merged.getArgument(i));
  }
  for (unsigned i = 0; i < nSecondArgs; ++i) {
    mapper.map(second.getArgument(i), merged.getArgument(nFirstArgs + i));
  }
  // clone body
  if (first.getBlocks().size() != 1 || second.getBlocks().size() != 1) {
    // TODO: support unstructured control flow in kernels
    return nullptr;
  }
  Block &firstBody = first.getBody().front();
  for (auto &op : firstBody.getOperations()) {
    if (isa<allo::ReturnOp>(op))
      continue;
    b.clone(op, mapper);
  }
  Block &secondBody = second.getBody().front();
  for (auto &op : secondBody.getOperations()) {
    if (isa<allo::ReturnOp>(op))
      continue;
    b.clone(op, mapper);
  }
  allo::ReturnOp::create(b, first.getLoc(), ValueRange{});
  return merged;
}

static DenseMap<Block *, SmallVector<CallOpInterface, 4>>
groupAndSortCallsByBlock(ArrayRef<CallOpInterface> ops) {
  DenseMap<Block *, SmallVector<CallOpInterface, 4>> blockToCalls;
  for (auto op : ops) {
    blockToCalls[op->getBlock()].push_back(op);
  }
  for (auto &[_, calls] : blockToCalls) {
    llvm::sort(calls, [](CallOpInterface a, CallOpInterface b) {
      return a->isBeforeInBlock(b);
    });
  }
  return blockToCalls;
}

static LogicalResult replaceCallsToNewKernel(RewriterBase &b,
                                             ArrayRef<CallOpInterface> opsA,
                                             ArrayRef<CallOpInterface> opsB,
                                             StringRef kernelName) {
  OpBuilder::InsertionGuard g(b);
  auto groupedA = groupAndSortCallsByBlock(opsA);
  auto groupedB = groupAndSortCallsByBlock(opsB);

  // verify the calls have the same block-level grouping.
  if (groupedA.size() != groupedB.size())
    return failure();
  for (auto &[block, callAs] : groupedA) {
    auto it = groupedB.find(block);
    if (it == groupedB.end())
      return failure();
    if (callAs.size() != it->second.size())
      return failure();
  }
  for (auto &[block, callAs] : groupedA) {
    auto &callBs = groupedB.find(block)->second;
    for (auto [callA, callB] : llvm::zip(callAs, callBs)) {
      auto later = callA->isBeforeInBlock(callB) ? callB : callA;
      b.setInsertionPoint(later);

      SmallVector<Value, 4> operands;
      llvm::append_range(operands, callA->getOperands());
      llvm::append_range(operands, callB->getOperands());
      SmallVector<Type, 4> resTypes;
      llvm::append_range(resTypes, callA->getResultTypes());
      llvm::append_range(resTypes, callB->getResultTypes());
      auto call = allo::CallOp::create(b, later.getLoc(), kernelName, resTypes,
                                       operands);
      auto newResults = call.getResults();
      auto aNumResults = callA->getNumResults();
      b.replaceAllUsesWith(callA->getResults(),
                           newResults.take_front(aNumResults));
      b.replaceAllUsesWith(callB->getResults(),
                           newResults.drop_front(aNumResults));

      b.eraseOp(callA);
      b.eraseOp(callB);
    }
  }
  return success();
}

static LogicalResult applyForwardingOnMergedKernel(
    RewriterBase &b, const DominanceInfo &dom,
    const DenseMap<Operation *, Operation *> &opMap,
    SmallVectorImpl<ChannelForwardPair> &matchedPairs) {
  OpBuilder::InsertionGuard g(b);
  for (auto &pair : matchedPairs) {
    Operation *clonedWriteOp = opMap.lookup(pair.writeOp);
    Operation *clonedReadOp = opMap.lookup(pair.readOp);
    assert(clonedReadOp && clonedWriteOp &&
           "expected matched ops to be cloned into merged kernel");
    auto put = dyn_cast<ChanPutOp>(clonedWriteOp);
    auto get = dyn_cast<ChanGetOp>(clonedReadOp);
    auto acq = dyn_cast<ChanAcquireOp>(clonedReadOp);
    auto rel = dyn_cast<ChanReleaseOp>(clonedWriteOp);
    if (put && get) {
      if (!dom.dominates(put.getValue(), get)) {
        InFlightDiagnostic diag = emitRemark(put.getLoc());
        diag << "forwarding chan.put does not dominate chan.get in merged "
                "kernel";
        diag.attachNote(get.getLoc()) << "chan.get here";
        return failure();
      }
      // perform forwarding
      b.replaceAllUsesWith(get.getValue(), put.getValue());
      b.eraseOp(get);
      b.eraseOp(put);
    } else if (acq && rel) {
      if (!dom.dominates(rel, acq)) {
        InFlightDiagnostic diag = emitRemark(rel.getLoc());
        diag << "forwarded release does not dominate acquire in merged kernel";
        diag.attachNote(acq.getLoc()) << "acquire here";
        return failure();
      }
      // perform forwarding
      b.replaceAllUsesWith(acq.getBuffers(), rel.getBuffers());
      b.eraseOp(acq);
      b.eraseOp(rel);
    } else {
      llvm_unreachable(
          "unexpected matched pair of channel access ops for forwarding");
    }
  }
  return success();
}

DiagnosedSilenceableFailure
transform::ChainOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {
  // Precondition checks
  auto kernels = llvm::to_vector(state.getPayloadOps(getKernels()));
  if (llvm::any_of(kernels,
                   [](Operation *op) { return !isa<allo::KernelOp>(op); })) {
    return emitSilenceableError()
           << "chain requires all payload ops in the input handle to be "
              "allo.kernel";
  }
  if (kernels.size() < 2) {
    return emitSilenceableError()
           << "chain requires at least two kernels as input";
  }
  auto mod = kernels.front()->getParentOfType<ModuleOp>();
  if (!llvm::all_of(kernels, [&](Operation *op) {
        return op->getParentOfType<ModuleOp>() == mod;
      })) {
    return emitSilenceableError()
           << "chain requires kernels in the same module";
  }

  DataflowGraph graph(mod);
  graph.init();

  auto current = cast<KernelOp>(kernels.front());
  for (unsigned i = 1; i < kernels.size(); ++i) {
    auto next = cast<KernelOp>(kernels[i]);
    // check if violate data dependence
    if (graph.hasBackwardDependence(current, next)) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableFailure(current)
          << "cannot chain kernels with backward dataflow dependence: @"
          << current.getSymName() << " and @" << next.getSymName();
      diag.attachNote(next.getLoc()) << "conflicting kernel here";
      return diag;
    }
    // merge kernel bodies
    IRMapping mapper;
    auto merged = createMergedKernel(rewriter, current, next, mapper);
    if (!merged) {
      DiagnosedSilenceableFailure diag = emitSilenceableFailure(current);
      diag << "cannot resolve attributes/arguments conflicts when merging "
              "kernels";
      diag.attachNote(next.getLoc()) << "conflicting kernel here";
      return diag;
    }
    DenseSet<StringAttr> channels;
    for (auto *dep : graph.getEdge(current, next))
      channels.insert(dep->channel);

    for (auto channelAttr : llvm::make_early_inc_range(channels)) {
      // get analysis of the chan.put/get pair
      auto fwdPairsOr = graph.analyzePutGetForward(current, next, channelAttr,
                                                   /*emitError=*/true);
      if (failed(fwdPairsOr)) {
        InFlightDiagnostic diag = emitWarning()
                                  << "skipping forwarding for channel @"
                                  << channelAttr.getValue();
        continue;
      }
      DominanceInfo dom(merged);
      if (failed(applyForwardingOnMergedKernel(
              rewriter, dom, mapper.getOperationMap(), *fwdPairsOr))) {
        continue;
      }
      channels.erase(channelAttr);
    }

    // rewrite calls to current and next to call the merged kernel
    auto currUsesOr = SymbolTable::getSymbolUses(current.getSymNameAttr(), mod);
    auto nextUsesOr = SymbolTable::getSymbolUses(next.getSymNameAttr(), mod);
    if (!currUsesOr || !nextUsesOr) {
      current = merged;
      continue;
    }
    SmallVector<CallOpInterface, 4> currCalls, nextCalls;
    for (auto use : *currUsesOr) {
      auto call = dyn_cast<CallOpInterface>(use.getUser());
      assert(call);
      currCalls.push_back(call);
    }
    for (auto use : *nextUsesOr) {
      auto call = dyn_cast<CallOpInterface>(use.getUser());
      assert(call);
      nextCalls.push_back(call);
    }
    if (failed(replaceCallsToNewKernel(rewriter, currCalls, nextCalls,
                                       merged.getSymName()))) {
      return emitSilenceableFailure(current)
             << "cannot rewrite calls to merged kernel";
    }
    if (failed(graph.mergeNodes(current, next, merged, channels))) {
      return emitSilenceableFailure(current)
             << "dataflow graph update failed after merging kernels";
    }
    rewriter.eraseOp(current);
    rewriter.eraseOp(next);
    current = merged;
  }

  results.set(cast<OpResult>(getResult()), {current.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

///===--------------------------------------------------------------------===///
/// BundleOp
///===--------------------------------------------------------------------===///
DiagnosedSilenceableFailure
transform::BundleOp::apply(transform::TransformRewriter &rewriter,
                           transform::TransformResults &results,
                           transform::TransformState &state) {
  return emitSilenceableError() << "transform.bundle is not implemented yet";
}
