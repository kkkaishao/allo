/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h" // kPartitionAttr
#include "allo/IR/AlloAttrs.h"
#include "allo/Scheduling/MemoryModel.h" // joinPartitions, bankLayoutOf
#include "allo/Scheduling/RegionGraph.h" // buildAndSortCallsiteGraph
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::allo {
#define GEN_PASS_DEF_PROPAGATEPARTITIONPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// Where an array's `allo.part` lives: the op that allocates it, the
// `memref.global` that declares it, or a function's argument attributes. One
// physical array has one carrier per function it is visible in. Mirrors
// `MemoryModel`'s `carrierAttr`, which is how the scheduler and the emitter
// read the attribute back.
struct Carrier {
  Operation *op;                     // alloc / alloca / memref.global / func
  std::optional<unsigned> argNumber; // set iff `op` is the func owning the arg
  MemRefType type;

  PartitionAttr part() {
    if (argNumber)
      return cast<func::FuncOp>(op).getArgAttrOfType<PartitionAttr>(
          *argNumber, kPartitionAttr);
    return op->getAttrOfType<PartitionAttr>(kPartitionAttr);
  }
  void setPart(PartitionAttr p) {
    if (argNumber)
      cast<func::FuncOp>(op).setArgAttr(*argNumber, kPartitionAttr, p);
    else
      op->setAttr(kPartitionAttr, p);
  }
};

// The carrier \p memref resolves to, or nullopt when this pass cannot name one.
// A rank-preserving `memref.cast` is followed, since `allo.part` names
// dimensions and a cast leaves them lined up; a view that RESHAPES is not, as
// its dimensions are not the underlying array's.
static std::optional<Carrier> carrierOf(Value memref) {
  while (auto castOp = memref.getDefiningOp<memref::CastOp>()) {
    auto from = dyn_cast<MemRefType>(castOp.getSource().getType());
    auto to = dyn_cast<MemRefType>(castOp.getType());
    if (!from || !to || from.getRank() != to.getRank())
      break;
    memref = castOp.getSource();
  }
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type)
    return std::nullopt;
  if (auto arg = dyn_cast<BlockArgument>(memref)) {
    auto fn = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
    if (!fn)
      return std::nullopt;
    return Carrier{fn, arg.getArgNumber(), type};
  }
  Operation *def = memref.getDefiningOp();
  if (auto get = dyn_cast<memref::GetGlobalOp>(def)) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        get, get.getNameAttr());
    assert(global && "get_global references an undefined memref.global");
    return Carrier{global, std::nullopt, type};
  }
  if (!isa<memref::AllocOp, memref::AllocaOp>(def))
    return std::nullopt;
  return Carrier{def, std::nullopt, type};
}

// How a carrier reads in a diagnostic: the array it stands for, named the way
// the user wrote it.
static std::string describeCarrier(Carrier &c) {
  std::string out;
  llvm::raw_string_ostream os(out);
  if (auto fn = dyn_cast<func::FuncOp>(c.op))
    os << "argument " << *c.argNumber << " of kernel '" << fn.getName() << "'";
  else if (auto global = dyn_cast<memref::GlobalOp>(c.op))
    os << "global array '" << global.getSymName() << "'";
  else
    os << "array allocated at " << c.op->getLoc();
  return out;
}

// Union-find over carriers. Two carriers are the SAME STORAGE when a call
// passes one as the other's argument: a sub-kernel MASTERS PORTS on the
// caller's array rather than receiving a copy. Two callsites handing different
// arrays to one callee unify those arrays too, which is the transitive closure
// the callee's single body demands.
struct Storage {
  SmallVector<Carrier> carriers;
  SmallVector<unsigned> parent;
  DenseMap<std::pair<Operation *, int>, unsigned> index;

  unsigned add(Carrier &c) {
    auto key = std::pair{c.op, c.argNumber ? int(*c.argNumber) : -1};
    auto [slot, fresh] = index.try_emplace(key, carriers.size());
    if (fresh) {
      carriers.push_back(c);
      parent.push_back(slot->second);
    }
    return slot->second;
  }
  unsigned find(unsigned i) {
    while (parent[i] != i)
      i = parent[i] = parent[parent[i]];
    return i;
  }
  void unite(unsigned a, unsigned b) { parent[find(a)] = find(b); }
};

struct PropagatePartitionPass
    : public allo::impl::PropagatePartitionPassBase<PropagatePartitionPass> {
  using PropagatePartitionPassBase::PropagatePartitionPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    topFunc = module.lookupSymbol<func::FuncOp>(top);
    if (!topFunc) {
      error(Stage::Prep, module) << "Top function '" << top << "' not found";
      return signalPassFailure();
    }
    auto callsOr = buildAndSortCallsiteGraph(topFunc);
    if (failed(callsOr))
      return signalPassFailure();

    Storage storage;
    if (failed(unifyAcrossCalls(*callsOr, storage)))
      return signalPassFailure();

    // Group in discovery order: the diagnostics below name the carriers that
    // disagree, and a hashed iteration would leave WHICH one to the allocator.
    SmallVector<SmallVector<unsigned>> classes;
    DenseMap<unsigned, unsigned> classOf;
    for (unsigned i = 0, e = storage.carriers.size(); i < e; ++i) {
      auto [slot, fresh] = classOf.try_emplace(storage.find(i), classes.size());
      if (fresh)
        classes.emplace_back();
      classes[slot->second].push_back(i);
    }

    for (ArrayRef<unsigned> members : classes)
      if (failed(reconcile(storage, members)))
        return signalPassFailure();
  }

  // One carrier per array per function, unified along every call-argument edge.
  // An async spawn is a `func.call` like any other, so a dataflow process's
  // parameters join the same class its container's buffer is in.
  LogicalResult unifyAcrossCalls(ArrayRef<Operation *> calls,
                                 Storage &storage) {
    SymbolTableCollection syms;
    for (Operation *op : calls) {
      auto call = cast<func::CallOp>(op);
      auto callee = syms.lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isExternal())
        continue;
      for (auto [k, actual] : llvm::enumerate(call.getArgOperands())) {
        auto type = dyn_cast<MemRefType>(actual.getType());
        if (!type)
          continue;
        Carrier param{callee, unsigned(k), type};
        std::optional<Carrier> array = carrierOf(actual);
        if (array) {
          storage.unite(storage.add(*array), storage.add(param));
          continue;
        }
        // An array whose storage this pass cannot name propagates nothing; a
        // partition stated at one end has no carrier to reconcile the other on.
        BankLayout here = bankLayoutOf(actual);
        if (!param.part() && here.numBanks == 1 && !here.registers)
          continue;
        unsupported(Stage::Prep, call)
            << "Array argument " << k << " of sub-kernel '" << call.getCallee()
            << "' is partitioned but reaches the call through a view, whose "
               "banking is not the underlying array's; pass the array itself, "
               "or partition it where it is allocated";
        return failure();
      }
    }
    return success();
  }

  // Settle one physical array: join every partition stated on any of its
  // carriers, then write the result back to all of them. ONE sweep reaches the
  // fixpoint, since which carriers denote one array follows from the call graph
  // alone and never from the attributes, and the join is associative.
  LogicalResult reconcile(Storage &storage, ArrayRef<unsigned> members) {
    MemRefType type = storage.carriers[members.front()].type;
    PartitionAttr joined;
    for (unsigned m : members) {
      Carrier &c = storage.carriers[m];
      assert(c.type.getRank() == type.getRank() &&
             "a call argument and its parameter have the same memref type");
      PartitionAttr here = c.part();
      if (!here)
        continue;
      std::string why;
      auto next = joinPartitions(joined, here, type, why);
      if (failed(next)) {
        auto diag = error(Stage::Prep, c.op);
        diag << "Array partitioning conflict: " << describeCarrier(c) << " is "
             << here << ", which cannot be reconciled with the " << joined
             << " stated elsewhere on the same array, because " << why
             << ". One array has one layout, and a sub-kernel addresses it in "
                "the banks its caller allocated. The array is named by:";
        for (unsigned n : members)
          diag << "\n  -> " << describeCarrier(storage.carriers[n]);
        return failure();
      }
      joined = *next;
    }
    if (!joined)
      return success();

    for (unsigned m : members) {
      Carrier &c = storage.carriers[m];
      if (c.part() == joined)
        continue;
      // Reaching the top's arguments changes the DESIGN's boundary: each bank
      // becomes its own port group and the host shards the array by bank.
      if (c.op == topFunc.getOperation())
        info(Stage::Prep, topFunc)
            << "Argument " << *c.argNumber << " of the top kernel takes the "
            << joined
            << " its sub-kernels agree on, so the design's boundary now "
               "carries one port group per bank";
      c.setPart(joined);
    }
    return success();
  }

  func::FuncOp topFunc;
};

} // namespace
