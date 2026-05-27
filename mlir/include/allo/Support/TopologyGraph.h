#ifndef ALLO_SUPPORT_TOPOLOGYGRAPH_H
#define ALLO_SUPPORT_TOPOLOGYGRAPH_H

#include "allo/IR/AlloOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::allo {

constexpr llvm::StringLiteral kGridAttrName = "allo.spmd.grid";
constexpr llvm::StringLiteral kCoordAttrName = "allo.spmd.coord";

struct ProcessNode {
  unsigned id = 0;
  InvokeOp invoke;
  KernelOp parent; // the nearest ancestor kernel containing this node
  KernelOp callee;

  /// SPMD related information.
  // the mapping attribute of the callee
  SmallVector<int32_t, 4> mapping;
  // the grid that this node is mapped to
  SmallVector<int32_t, 4> grid;
  // the coordinate of this node in the grid
  SmallVector<int32_t, 4> coord;

  bool isConcrete = false;
};

struct Endpoint {
  enum class Kind { Producer, Consumer };
  // the id of the node that produces or consumes from the channel
  unsigned nodeId = 0;
  // the argument number of the stream in the invoke operation
  unsigned argNo = 0;
  Operation *streamOp = nullptr;
  Kind kind = Kind::Producer;
  SmallVector<int64_t, 4> lane;
};

struct Channel {
  Value stream;
  StreamType streamType;
  SmallVector<int64_t, 4> lane;
  SmallVector<Endpoint, 2> producers;
  SmallVector<Endpoint, 2> consumers;
  bool isSame(Value stream, ArrayRef<int64_t> lane) const {
    return this->stream == stream && this->lane.size() == lane.size() &&
           llvm::equal(this->lane, lane);
  }
};

struct TopologyGraph {
public:
  explicit TopologyGraph(KernelOp scope) : scope(scope) {};

  ArrayRef<ProcessNode> getNodes() const { return nodes; }
  ArrayRef<Channel> getChannels() const { return channels; }

  void exportAsDot(raw_ostream &os) const;
  std::string exportAsDot() const {
    std::string out;
    llvm::raw_string_ostream os(out);
    exportAsDot(os);
    return out;
  }

  unsigned addNode(InvokeOp invoke, KernelOp callee);
  LogicalResult addEndpoint(unsigned nodeId, InvokeOp invoke,
                            Operation *streamOp, Value stream,
                            Endpoint::Kind kind, bool skipDynamicLanes = false);
  Channel &getOrAddChannel(Value stream, ArrayRef<int64_t> lane);

  KernelOp scope;
  SmallVector<ProcessNode, 8> nodes;
  SmallVector<Channel, 8> channels;
};

FailureOr<TopologyGraph> buildTopologyGraph(KernelOp scope,
                                            SymbolTableCollection &symbols,
                                            bool skipDynamicLanes = false);

} // namespace mlir::allo

#endif // ALLO_SUPPORT_TOPOLOGYGRAPH_H
