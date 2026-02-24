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
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace mlir;
using namespace mlir::allo;

///===----------------------------------------------------------------------===//
/// ApplyVirtualMapOp
///===----------------------------------------------------------------------===//
namespace {
using Pid = SmallVector<int64_t, 4>;
using PidList = SmallVector<Pid, 8>;

void enumeratePids(ArrayRef<int64_t> mapping, unsigned dim, Pid &current,
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

std::string buildInstanceName(StringRef base, ArrayRef<int64_t> pid) {
  std::string name = base.str() + "__vm";
  for (int64_t v : pid)
    name += "_" + std::to_string(v);
  return name;
}
} // namespace

DiagnosedSilenceableFailure transform::ApplyVirtualMapOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto k = dyn_cast<allo::KernelOp>(target);
  if (!k) {
    return emitSilenceableFailure(target)
           << "expected allo.kernel op, got " << target->getName();
  }
  auto givenMapping = getMapping();
  ArrayRef<int64_t> mapping;
  // if given, use the mapping from the op attribute
  if (givenMapping.has_value()) {
    mapping = getMapping()->getRanges();
  } else {
    mapping = k.getVirtualMappingVec();
  }
  // if no mapping is needed, skip the transformation
  if (isTrivialMapping(mapping))
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
    enumeratePids(mapping, /*dim=*/0, current, pids);
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
    clone->setAttr(clone.getVirtualMappingAttrName(),
                   VirtMapAttr::get(rewriter.getContext(), 1));

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
                                                  mapping[dim]);
        rewriter.replaceAllOpUsesWith(nProgsOp, cst);
        opToErase.push_back(nProgsOp);
      }
    });
    for (auto op : opToErase)
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
    auto call = dyn_cast<allo::CallOp>(use.getUser());
    assert(call && "expected call op");
    rewriter.setInsertionPoint(call);
    for (const auto &inst : instNames) {
      allo::CallOp::create(rewriter, call.getLoc(), inst, call.getResultTypes(),
                           call.getOperands());
    }
    rewriter.eraseOp(call);
  }
  // erase the original kernel
  k.erase();
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===//
/// ChainOp
///===----------------------------------------------------------------------===//
namespace {
std::string makeMergedKernelName(StringRef first, StringRef second) {
  return first.str() + "-" + second.str();
}

allo::KernelOp createMergedEmptyKernel(RewriterBase &b, KernelOp first,
                                       KernelOp second, IRMapping &mapper) {
  OpBuilder::InsertionGuard g(b);
  // merge arguments
  auto argTypes = llvm::to_vector(first.getArgumentTypes());
  llvm::append_range(argTypes, second.getArgumentTypes());
  auto resTypes = second.getResultTypes();
  auto funcTy = FunctionType::get(b.getContext(), argTypes, resTypes);
  b.setInsertionPoint(first);
  auto funcName = makeMergedKernelName(first.getSymName(), second.getSymName());
  // merge attributes
  NamedAttrList attrList(first->getDiscardableAttrDictionary());
  for (auto attr : second->getDiscardableAttrDictionary()) {
    if (attrList.getNamed(attr.getName())) {
      return nullptr; // conflict, cannot merge
    }
  }
  // merge arg attributes
  SmallVector<DictionaryAttr, 8> argAttrs;
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
  auto merged = allo::KernelOp::create(b, first.getLoc(), funcName, funcTy,
                                       attrList, argAttrs, 1);
  Block *body = merged.addEntryBlock();
  b.setInsertionPointToEnd(body);
  allo::ReturnOp::create(b, first.getLoc(), {});
  // map from old args to new args
  unsigned nFirstArgs = first.getNumArguments();
  unsigned nSecondArgs = second.getNumArguments();
  for (unsigned i = 0; i < nFirstArgs; ++i) {
    mapper.map(first.getArgument(i), merged.getArgument(i));
  }
  for (unsigned i = 0; i < nSecondArgs; ++i) {
    mapper.map(second.getArgument(i), merged.getArgument(nFirstArgs + i));
  }
  return merged;
}
} // namespace

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
  if (llvm::all_of(kernels, [&](Operation *op) {
        return op->getParentOfType<ModuleOp>() == mod;
      })) {
    return emitSilenceableError()
           << "chain requires kernels in the same module";
  }
  for (unsigned i = 0; i < kernels.size() - 1; ++i) {
    // create fused kernel with empty body
    auto first = cast<allo::KernelOp>(kernels[i]);
    auto second = cast<allo::KernelOp>(kernels[i + 1]);
    IRMapping mapper;
    auto merged = createMergedEmptyKernel(rewriter, first, second, mapper);
    if (!merged) {
      DiagnosedSilenceableFailure diag = emitSilenceableFailure(first);
      diag << "cannot resolve attributes/arguments conflicts when merging "
              "kernels";
      diag.attachNote(second.getLoc()) << "conflicting kernel here";
      return diag;
    }
    // merge bodies to the new kernel
    Block &firstBody = first.getBody().front();
    rewriter.setInsertionPointToStart(&merged.getBody().front());
    for (auto &op : firstBody.getOperations()) {
      if (isa<allo::ReturnOp>(op))
        continue;
      Operation *cloned = rewriter.clone(op, mapper);
    }
    Block &secondBody = second.getBody().front();
    rewriter.setInsertionPointToStart(&merged.getBody().front());
    for (auto &op : secondBody.getOperations()) {
      if (isa<allo::ReturnOp>(op))
        continue;
      Operation *cloned = rewriter.clone(op, mapper);
    }
  }
  return DiagnosedSilenceableFailure::success();
}

///===--------------------------------------------------------------------===///
/// BundleOp
///===--------------------------------------------------------------------===///
DiagnosedSilenceableFailure
transform::BundleOp::apply(transform::TransformRewriter &rewriter,
                           transform::TransformResults &results,
                           transform::TransformState &state) {
  // TODO:
  return DiagnosedSilenceableFailure::success();
}
