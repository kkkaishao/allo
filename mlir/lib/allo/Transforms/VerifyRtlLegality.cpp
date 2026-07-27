/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/DependenceAnalysis.h" // isUnmodeledMemoryAccess
#include "allo/Scheduling/MemoryModel.h"
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/ProblemBuilder.h" // whileFlushingPipelines
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::allo {
#define GEN_PASS_DEF_VERIFYRTLLEGALITYPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// An op the reifier turns into a `dcp.compute`. It takes the IP path when a
// library row matched and the combinational path otherwise, so exactly these
// ops need a realization. Scoped by dialect, since only these are scheduled as
// compute: `ConvertScheduleToDcp` reaches the same split for a single-result
// non-constant op that carries a start time.
bool isComputeOp(Operation *op) {
  return op->getNumResults() == 1 && !op->hasTrait<OpTrait::ConstantLike>() &&
         (isa<arith::ArithDialect, math::MathDialect>(op->getDialect()) ||
          isa<affine::AffineApplyOp>(op));
}

// One end of a channel: which call holds it, and which way tokens move.
struct CallEnd {
  Operation *call;
  bool isInput; // the child GETS from the channel
};

// A channel as this function sees it: the ends it issues itself, the ends its
// children hold, and the seed that breaks a feedback cycle's start dependence.
struct Channel {
  Value root;
  bool internal = false; // declared here (`stream.create`) vs a boundary arg
  ArrayAttr init;
  SmallVector<Operation *> accesses; // this function's own get / put ops
  bool anyPut = false, anyGet = false;
  unsigned producers = 0; // CHILDREN writing it; a local put is not one
  SmallVector<CallEnd> callEnds;
};

struct VerifyRtlLegalityPass
    : public allo::impl::VerifyRtlLegalityPassBase<VerifyRtlLegalityPass> {
  using VerifyRtlLegalityPassBase::VerifyRtlLegalityPassBase;

  // Which way a callee's stream parameter carries tokens. A parameter accessed
  // in the callee takes the direction of its first access, matching how
  // `DatapathBuilder::getOrCreateStream` fixes a channel's direction; one only
  // passed further down inherits the grandchild's, so this is filled callee
  // before caller and read back here.
  llvm::DenseMap<std::pair<Operation *, unsigned>, bool> streamArgIsInput;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto topFunc = module.lookupSymbol<func::FuncOp>(top);
    if (!topFunc) {
      error(Stage::Prep, module) << "Top function '" << top << "' not found";
      return signalPassFailure();
    }
    auto orderOr = buildAndSortCallsiteGraph(topFunc);
    if (failed(orderOr))
      return signalPassFailure();

    // The closure the emit driver visits, callees before callers so a call can
    // read facts already computed for its callee.
    SetVector<Operation *> closure;
    SymbolTableCollection syms;
    for (Operation *op : *orderOr)
      if (auto callee = syms.lookupNearestSymbolFrom<func::FuncOp>(
              op, cast<func::CallOp>(op).getCalleeAttr());
          callee && !callee.isExternal())
        closure.insert(callee);
    closure.insert(topFunc);

    OperatorLibrary lib = OperatorLibrary::fromModule(module);
    for (Operation *op : closure)
      recordStreamArgDirections(cast<func::FuncOp>(op));
    for (Operation *op : closure)
      if (failed(verifyFunc(cast<func::FuncOp>(op), module, lib)))
        return signalPassFailure();
  }

  void recordStreamArgDirections(func::FuncOp func) {
    for (BlockArgument arg : func.getArguments()) {
      if (!isa<StreamType>(arg.getType()))
        continue;
      for (Operation *user : arg.getUsers()) {
        std::optional<bool> dir;
        if (isa<StreamGetOp>(user))
          dir = true;
        else if (isa<StreamPutOp>(user))
          dir = false;
        else if (auto call = dyn_cast<func::CallOp>(user))
          dir = calleeStreamDirection(call, arg);
        if (dir) {
          streamArgIsInput[{func.getOperation(), arg.getArgNumber()}] = *dir;
          break;
        }
      }
    }
  }

  // The direction \p call imposes on \p stream, from the callee parameter it
  // is passed to. Empty when the callee never resolves one, which the unused
  // boundary-argument check below reports against the callee itself.
  std::optional<bool> calleeStreamDirection(func::CallOp call, Value stream) {
    SymbolTableCollection syms;
    auto callee =
        syms.lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
    if (!callee)
      return std::nullopt;
    for (auto [k, actual] : llvm::enumerate(call.getArgOperands())) {
      if (actual != stream)
        continue;
      auto it = streamArgIsInput.find(
          {callee.getOperation(), static_cast<unsigned>(k)});
      if (it != streamArgIsInput.end())
        return it->second;
    }
    return std::nullopt;
  }

  LogicalResult verifyFunc(func::FuncOp func, ModuleOp module,
                           const OperatorLibrary &lib) {
    if (failed(checkSignature(func)) || failed(checkOperations(func, lib)) ||
        failed(checkMemories(func, lib.memoryLibrary())) ||
        failed(checkComposition(func, lib)))
      return failure();
    return checkChannels(func);
  }

  //===--------------------------------------------------------------------===//
  // Signature and operations.
  //===--------------------------------------------------------------------===//

  LogicalResult checkSignature(func::FuncOp func) {
    for (Type t : func.getResultTypes())
      if (isa<MemRefType>(t)) {
        unsupported(Stage::Prep, func)
            << "Returning a memref is not lowered yet; write the result "
               "through an output argument (out-parameter) instead";
        return failure();
      }
    return success();
  }

  LogicalResult checkOperations(func::FuncOp func, const OperatorLibrary &lib) {
    WalkResult r = func.walk([&](Operation *op) {
      if (isUnmodeledMemoryAccess(op)) {
        unsupported(Stage::Prep, op)
            << "Operation '" << op->getName()
            << "' carries a memory effect the dependence analysis does not "
               "model, so scheduling would reorder it against the accesses it "
               "aliases. A whole-array assignment (`buf = A`) lowers to "
               "`memref.copy`: write the array element by element in a loop "
               "instead";
        return WalkResult::interrupt();
      }
      if (!isComputeOp(op))
        return WalkResult::advance();
      std::string symbol = lib.lookup(op).symbol;
      // The two realization paths, in the order the reifier tries them: an IP
      // row's symbol, else a native comb lowering.
      if (symbol.empty() && !combKindOf(op)) {
        error(Stage::Prep, op)
            << "Operator '" << op->getName()
            << "' is not realized by the device: it has neither an IP module "
               "nor a native lowering. Declare an @ip for it, or add native "
               "support";
        return WalkResult::interrupt();
      }
      if (!symbol.empty() && failed(checkStallContract(op, symbol)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return failure(r.wasInterrupted());
  }

  // `ce` is the only IP port ABI the emitter realizes. `elastic` is
  // variable-latency, which nothing downstream honors: consumers are scheduled
  // at the operator's fixed latency and the instance gets the free-running port
  // shape.
  LogicalResult checkStallContract(Operation *op, StringRef symbol) {
    auto opr = SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(
        op, StringAttr::get(op->getContext(), symbol));
    assert(opr && "a matched operator row names a live dcp.operator");
    if (opr.getStall() != StallContractEnum::Elastic)
      return success();
    error(Stage::Prep, op)
        << "Operator IP '" << symbol
        << "' declares the elastic (valid/ready, variable-latency) stall "
           "contract, which is not realized. Declare style='ce'";
    return failure();
  }

  //===--------------------------------------------------------------------===//
  // Storage.
  //===--------------------------------------------------------------------===//

  LogicalResult checkMemories(func::FuncOp func, const MemoryLibrary &memLib) {
    SmallVector<Value> arrays;
    for (BlockArgument arg : func.getArguments())
      if (isa<MemRefType>(arg.getType()))
        arrays.push_back(arg);
    func.walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(op))
        arrays.push_back(op->getResult(0));
    });

    for (Value array : arrays) {
      MemoryImplEnum impl = characterize(array, memLib.defaultImpl).impl;
      Operation *anchor =
          array.getDefiningOp() ? array.getDefiningOp() : func.getOperation();
      // An implementation the device never declared would fall to the
      // zero-timing default and schedule combinationally, reading before valid.
      if (!memLib.declares(impl)) {
        error(Stage::Prep, anchor)
            << "No memory characterization for storage impl '"
            << stringifyMemoryImplEnum(impl)
            << "'; declare it in the device `memory` table";
        return failure();
      }
      RWLatency lat = memLib.timing(impl).latency;
      // A boundary array's port latency is a contract with the driver, not
      // enforced by the emitted RTL, so any latency >= 1 works; 0 does not,
      // since an edge-triggered port cannot.
      if (isa<BlockArgument>(array) && (lat.read < 1 || lat.write < 1)) {
        error(Stage::Prep, func)
            << "Argument array with a " << lat.read << "-cycle read / "
            << lat.write
            << "-cycle write cannot be realized; a boundary port is "
               "edge-triggered and needs at least 1 cycle. Use an internal "
               "buffer, or bind this argument to a storage impl with a >= 1 "
               "cycle access";
        return failure();
      }
      // An internal array lives in an `seq.hlmem`, whose write is edge-
      // triggered too: a store commits at `writeLatency - 1`, which a 0-cycle
      // write wraps. A 0-cycle read is fine internally.
      if (lat.write < 1) {
        error(Stage::Prep, anchor)
            << "Storage impl '" << stringifyMemoryImplEnum(impl)
            << "' declares a 0-cycle write, which no array can be realized at: "
               "a write needs a clock edge to commit on. Give that row a write "
               "latency of at least 1 in the device `memory` table";
        return failure();
      }
    }
    return checkPartitionAgreement(func);
  }

  // A sub-kernel masters one port group per bank and indexes each in that
  // bank's own element space, so caller and callee must agree on the partition
  // factor; at a different factor the child addresses the wrong elements.
  // `propagate-partition` has already pushed a caller's partition down, so a
  // disagreement left here is one the callee declared for itself.
  LogicalResult checkPartitionAgreement(func::FuncOp func) {
    SymbolTableCollection syms;
    WalkResult r = func.walk([&](func::CallOp call) {
      auto callee = syms.lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isExternal())
        return WalkResult::advance();
      for (auto [k, actual] : llvm::enumerate(call.getArgOperands())) {
        if (!isa<MemRefType>(actual.getType()) || isa<BlockArgument>(actual) ||
            isConstantTable(actual))
          continue;
        unsigned here = bankLayoutOf(actual).numBanks;
        unsigned there = bankLayoutOf(callee.getArgument(k)).numBanks;
        if (here == there)
          continue;
        error(Stage::Prep, call)
            << "Array argument " << k << " of sub-kernel '" << call.getCallee()
            << "' is partitioned into " << there << " bank(s) there but into "
            << here
            << " in the caller; a sub-kernel addresses each bank in that "
               "bank's own space, so the two partitions must match. Give the "
               "array the same partition factor in both kernels";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(r.wasInterrupted());
  }

  //===--------------------------------------------------------------------===//
  // Composition and control shape.
  //===--------------------------------------------------------------------===//

  LogicalResult checkComposition(func::FuncOp func,
                                 const OperatorLibrary &lib) {
    if (composesOnStructuralTop(func)) {
      // A loop around a spawn reads as loose control flow to the check below,
      // so name it first: the loop is what the user has to move, not the
      // container's shape.
      WalkResult r = func.walk([&](func::CallOp call) {
        if (!call->getParentOfType<LoopLikeOpInterface>())
          return WalkResult::advance();
        error(Stage::Prep, call)
            << "A dataflow process is spawned inside a loop; a process is "
               "instantiated once and runs concurrently, so spawn it once and "
               "let it iterate internally (move the loop into the process)";
        return WalkResult::interrupt();
      });
      if (r.wasInterrupted())
        return failure();
      // `outline-loose-processes` lifts a loose span into a process of its own,
      // but skips a span whose live-in is not an array, a stream or a scalar,
      // and leaves the report to here.
      for (Operation &op : func.front())
        if (!isContainerStructure(op)) {
          unsupported(Stage::Prep, &op)
              << "A dataflow container with its own datapath (loose "
                 "load/store/compute beside the process network) is not "
                 "lowered yet; it composes child instances and channels only. "
                 "The outliner leaves a span in place when a value crossing it "
                 "is neither an array, a stream nor a scalar";
          return failure();
        }
    }

    WalkResult r = func.walk([&](scf::WhileOp w) {
      if (!whileFlushingPipelines(w, lib) || whileHasIdentityForwarding(w))
        return WalkResult::advance();
      error(Stage::Prep, w)
          << "While loop not scheduled: its loop-carried values are not "
             "forwarded 1:1 from the before-region through `scf.condition` "
             "into the after-region (they are reordered, dropped, or "
             "recombined), which the flushing-pipeline schedule requires; "
             "carry each value through unchanged";
      return WalkResult::interrupt();
    });
    return failure(r.wasInterrupted());
  }

  //===--------------------------------------------------------------------===//
  // Channels.
  //===--------------------------------------------------------------------===//

  // A channel is one {data,valid,ready} triple time-shared by every access to
  // it, and a directed cycle of unseeded channels deadlocks. Both properties
  // are settled by the process network's shape, which nothing between here and
  // emission changes.
  LogicalResult checkChannels(func::FuncOp func) {
    SmallVector<Channel> channels;
    llvm::DenseMap<Value, unsigned> index;
    auto channelFor = [&](Value stream) -> Channel & {
      auto [it, fresh] = index.try_emplace(stream, channels.size());
      if (fresh) {
        Channel ch;
        ch.root = stream;
        if (auto cr = stream.getDefiningOp<StreamCreateOp>()) {
          ch.internal = true;
          ch.init = cr.getInitAttr();
        }
        channels.push_back(std::move(ch));
      }
      return channels[it->second];
    };
    for (BlockArgument arg : func.getArguments())
      if (isa<StreamType>(arg.getType()))
        channelFor(arg);
    func.walk([&](StreamCreateOp cr) { channelFor(cr.getStream()); });

    func.walk([&](Operation *op) {
      if (isa<StreamGetOp, StreamPutOp>(op)) {
        Channel &ch = channelFor(op->getOperand(0));
        ch.accesses.push_back(op);
        (isa<StreamPutOp>(op) ? ch.anyPut : ch.anyGet) = true;
      } else if (auto call = dyn_cast<func::CallOp>(op)) {
        for (Value actual : call.getArgOperands()) {
          if (!isa<StreamType>(actual.getType()))
            continue;
          std::optional<bool> reads = calleeStreamDirection(call, actual);
          if (!reads)
            continue;
          Channel &ch = channelFor(actual);
          (*reads ? ch.anyGet : ch.anyPut) = true;
          ch.producers += !*reads;
          ch.callEnds.push_back({call, *reads});
        }
      }
    });

    for (const Channel &ch : channels)
      if (failed(checkChannelEnds(func, ch)))
        return failure();
    return checkChannelCycles(func, channels);
  }

  LogicalResult checkChannelEnds(func::FuncOp func, const Channel &ch) {
    // An access this module issues, else the child instance holding one of the
    // channel's ends. A boundary channel with neither has only the function.
    Operation *anchor = func.getOperation();
    if (!ch.accesses.empty())
      anchor = ch.accesses.front();
    else if (!ch.callEnds.empty())
      anchor = ch.callEnds.front().call;

    // Several READERS are a fan-out the emitter inserts (one FIFO each);
    // several WRITERS are a merge, whose token interleaving is not
    // deterministic.
    if (ch.producers > 1) {
      unsupported(Stage::Prep, anchor)
          << "A stream channel is written by more than one process; a channel "
             "is single-producer and a deterministic merge is not lowered yet";
      return failure();
    }
    // A port is an input or an output, so a boundary channel both read and
    // written has nothing to lower to.
    if (ch.anyPut && ch.anyGet && !ch.internal) {
      unsupported(Stage::Prep, anchor)
          << "A stream ARGUMENT both read and written inside one kernel is not "
             "lowered yet (a boundary channel lowers to one directional port); "
             "route the feedback through a second channel, or declare the "
             "channel inside the kernel";
      return failure();
    }
    // A local channel with one end only is a stall by construction: the puts
    // fill it and block, or the first get waits on a token nothing produces.
    if (ch.internal && !(ch.anyPut && ch.anyGet)) {
      error(Stage::Prep, anchor)
          << "The kernel-local stream is "
          << (ch.anyPut ? "never read" : "never written")
          << "; a channel needs both ends inside the kernel that owns it";
      return failure();
    }
    // A boundary argument nothing touches would leave a port undriven.
    if (!ch.internal && !ch.anyPut && !ch.anyGet) {
      error(Stage::Prep, anchor) << "The stream argument is neither read nor "
                                    "written";
      return failure();
    }
    return success();
  }

  // A directed cycle of channels with no initial tokens deadlocks, so it
  // suffices that the graph of UNSEEDED channels is acyclic. Insufficient
  // seeding (fewer tokens than the recurrence distance) surfaces as a hang.
  LogicalResult checkChannelCycles(func::FuncOp func,
                                   ArrayRef<Channel> channels) {
    llvm::DenseMap<Operation *, SmallVector<Operation *>> adj;
    SetVector<Operation *> nodes;
    for (const Channel &ch : channels) {
      for (const CallEnd &e : ch.callEnds)
        nodes.insert(e.call);
      if (ch.init && !ch.init.empty())
        continue;
      Operation *prod = nullptr;
      for (const CallEnd &e : ch.callEnds)
        if (!e.isInput)
          prod = e.call;
      if (!prod)
        continue; // fed from a boundary port: not part of a cycle
      for (const CallEnd &e : ch.callEnds)
        if (e.isInput)
          adj[prod].push_back(e.call);
    }

    llvm::DenseMap<Operation *, int> color; // 0 white / 1 gray / 2 black
    llvm::DenseMap<Operation *, Operation *> parent;
    SmallVector<Operation *> cycle;
    // Self-parameter recursive lambda (`self(self, ...)`): a local DFS with no
    // std::function type-erasure.
    auto visit = [&](auto &self, Operation *u) -> bool {
      color[u] = 1;
      for (Operation *v : adj[u]) {
        if (color[v] == 1) { // back edge -> the cycle v .. u -> v
          for (Operation *x = u; x != v; x = parent[x])
            cycle.push_back(x);
          cycle.push_back(v);
          return true;
        }
        if (color[v] == 0) {
          parent[v] = u;
          if (self(self, v))
            return true;
        }
      }
      color[u] = 2;
      return false;
    };
    for (Operation *n : nodes)
      if (cycle.empty() && color[n] == 0)
        visit(visit, n);
    if (cycle.empty())
      return success();

    std::reverse(cycle.begin(), cycle.end()); // producer order
    std::string path;
    llvm::raw_string_ostream os(path);
    for (Operation *x : cycle)
      os << cast<func::CallOp>(x).getCallee() << " -> ";
    os << cast<func::CallOp>(cycle.front()).getCallee(); // close the loop
    error(Stage::Prep, func)
        << "Dataflow feedback cycle [" << path
        << "] has no initial tokens and will deadlock; seed a channel on the "
           "cycle with an initializer, e.g. `s: Stream[T, depth] = [<init>]`";
    return failure();
  }
};

} // namespace
