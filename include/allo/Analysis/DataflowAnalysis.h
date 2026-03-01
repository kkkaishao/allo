#ifndef ALLO_DATAFLOW_ANALYSIS_H
#define ALLO_DATAFLOW_ANALYSIS_H

#include "allo/IR/AlloOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <string>
#include <utility>

namespace mlir::allo {

enum class ChannelPatternKind { None, SPSC, SPMC, MPSC, MPMC };

struct ChannelPatternInfo {
  ChannelPatternKind kind = ChannelPatternKind::None;
  ArrayRef<ChannelAccessOpInterface> writeOps;
  ArrayRef<ChannelAccessOpInterface> readOps;
};

struct ChannelAccessPoints {
  SetVector<ChannelAccessOpInterface> writeOps;
  SetVector<ChannelAccessOpInterface> readOps;
};

enum class ChannelAccessKind { Put, Get, Acquire, Release };

struct ChannelForwardPair {
  ChannelAccessOpInterface writeOp;
  ChannelAccessOpInterface readOp;
};

enum class KernelIsoKind { Iso, DifferOnlyByConstants, NotIso };

struct DataflowNode;
struct DataflowEdge {
  StringAttr channel;
  ChannelAccessOpInterface srcOp = nullptr;
  ChannelAccessOpInterface dstOp = nullptr;
  DataflowNode *srcNode = nullptr;
  DataflowNode *dstNode = nullptr;
};

struct DataflowNode {
  explicit DataflowNode(KernelOp kernel) : kernel(kernel) {}
  allo::KernelOp kernel;
};

struct DataflowGraph {
  using KernelPair = std::pair<KernelOp, KernelOp>;
  using DataflowEdgeList = SmallVector<DataflowEdge *, 4>;

  ModuleOp mod;
  MLIRContext *ctx;
  // hold the nodes and edges
  SmallVector<std::unique_ptr<DataflowNode>> nodes;
  SmallVector<std::unique_ptr<DataflowEdge>> edges;
  DenseMap<KernelOp, DataflowNode *> nodeMap;
  DenseMap<DataflowNode *, DataflowEdgeList> inEdges;
  DenseMap<DataflowNode *, DataflowEdgeList> outEdges;
  DenseMap<StringAttr, ChannelAccessPoints> channelAccesses;

  explicit DataflowGraph(ModuleOp mod) : mod(mod), ctx(mod.getContext()) {}

  void init();
  DataflowNode *addNode(KernelOp kernel);
  DataflowEdge *addEdge(StringAttr channel, ChannelAccessOpInterface srcOp,
                        ChannelAccessOpInterface dstOp);
  DataflowEdge *addEdge(StringRef channel, ChannelAccessOpInterface srcOp,
                        ChannelAccessOpInterface dstOp) {
    return addEdge(StringAttr::get(ctx, channel), srcOp, dstOp);
  }
  LogicalResult removeNode(DataflowNode *node);
  LogicalResult removeEdge(DataflowEdge *edge);
  DataflowEdgeList getEdge(KernelOp src, KernelOp dst) const;
  LogicalResult mergeNodes(KernelOp nodeA, KernelOp nodeB,
                           KernelOp mergedKernel,
                           DenseSet<StringAttr> &ignores);

  bool hasForwardDependence(KernelOp src, KernelOp dst) const;
  bool hasForwardDependence(KernelOp src, KernelOp dst,
                            StringRef channel) const;
  bool hasBackwardDependence(KernelOp src, KernelOp dst) const;
  bool hasBackwardDependence(KernelOp src, KernelOp dst,
                             StringRef channel) const;
  bool hasDependence(KernelOp lhs, KernelOp rhs) const;
  ChannelPatternInfo analyzeChannelPattern(StringAttr channel) const;
  ChannelPatternInfo analyzeChannelPattern(StringRef channel) const {
    return analyzeChannelPattern(StringAttr::get(ctx, channel));
  }
  static FailureOr<std::string>
  getAccessSiteSignature(ChannelAccessOpInterface op);
  FailureOr<SmallVector<ChannelForwardPair, 4>>
  analyzePutGetForward(KernelOp producer, KernelOp consumer, StringAttr channel,
                       bool emitError = false) const;
  bool hasCycle() const;

private:
  StringAttr getChannelAttr(StringRef channel) const;
  static ChannelAccessKind getChannelAccessKind(ChannelAccessOpInterface op);
  static bool isPutGetOp(Operation *op);
  static bool isChannelWriteOp(ChannelAccessOpInterface op);
  static bool isChannelReadOp(ChannelAccessOpInterface op);
};
} // namespace mlir::allo

#endif // ALLO_DATAFLOW_ANALYSIS_H
