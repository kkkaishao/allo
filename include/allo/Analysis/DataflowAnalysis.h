#ifndef ALLO_DATAFLOW_ANALYSIS_H
#define ALLO_DATAFLOW_ANALYSIS_H

#include "allo/IR/AlloOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir::allo {

enum class ChannelPatternKind { None, SPSC, SPMC, MPSC, MPMC };

struct ChannelPatternInfo {
  ChannelPatternKind Kind = ChannelPatternKind::None;
  ArrayRef<ChannelAccessOpInterface> WriteOps;
  ArrayRef<ChannelAccessOpInterface> ReadOps;
};

struct ChannelAccessPoints {
  SetVector<ChannelAccessOpInterface> WriteOps;
  SetVector<ChannelAccessOpInterface> ReadOps;
};

enum class ChannelAccessKind { Put, Get, Acquire, Release };

struct ChannelForwardPair {
  ChannelAccessOpInterface WriteOp;
  ChannelAccessOpInterface ReadOp;
};

enum class KernelIsoKind { Iso, DifferOnlyByConstants, NotIso };

struct DataflowNode;
struct DataflowEdge {
  StringAttr Channel;
  ChannelAccessOpInterface SrcOp = nullptr;
  ChannelAccessOpInterface DstOp = nullptr;
  DataflowNode *SrcNode = nullptr;
  DataflowNode *DstNode = nullptr;
};

struct DataflowNode {
  explicit DataflowNode(KernelOp kernel) : Kernel(kernel) {}
  allo::KernelOp Kernel;
};

struct DataflowGraph {
  using KernelPair = std::pair<KernelOp, KernelOp>;
  using DataflowEdgeList = SmallVector<DataflowEdge *, 4>;

  ModuleOp Mod;
  MLIRContext *Ctx;
  // hold the nodes and edges
  SmallVector<std::unique_ptr<DataflowNode>> Nodes;
  SmallVector<std::unique_ptr<DataflowEdge>> Edges;
  DenseMap<KernelOp, DataflowNode *> NodeMap;
  DenseMap<DataflowNode *, DataflowEdgeList> InEdges;
  DenseMap<DataflowNode *, DataflowEdgeList> OutEdges;
  DenseMap<StringAttr, ChannelAccessPoints> ChannelAccesses;

  explicit DataflowGraph(ModuleOp mod) : Mod(mod), Ctx(mod.getContext()) {}

  void init();
  DataflowNode *addNode(KernelOp kernel);
  DataflowEdge *addEdge(StringAttr channel, ChannelAccessOpInterface srcOp,
                        ChannelAccessOpInterface dstOp);
  DataflowEdge *addEdge(StringRef channel, ChannelAccessOpInterface srcOp,
                        ChannelAccessOpInterface dstOp) {
    return addEdge(StringAttr::get(Ctx, channel), srcOp, dstOp);
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
    return analyzeChannelPattern(StringAttr::get(Ctx, channel));
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
