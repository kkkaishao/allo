#include "allo/Support/TopologyGraph.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::allo;

static FailureOr<SmallVector<int64_t, 4>> getStaticLane(ValueRange indices) {
  SmallVector<int64_t, 4> lane;
  for (Value index : indices) {
    IntegerAttr::ValueType cst;
    if (!matchPattern(index, m_ConstantInt(&cst)))
      return failure();
    lane.push_back(cst.getSExtValue());
  }
  return lane;
}

static void printI32Array(raw_ostream &os, ArrayRef<int32_t> values) {
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";
}

static void printI64Array(raw_ostream &os, ArrayRef<int64_t> values) {
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";
}

static std::string stringifyType(Type type) {
  std::string out;
  llvm::raw_string_ostream os(out);
  type.print(os);
  return out;
}

static std::string stringifyValue(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << "arg" << arg.getArgNumber();
    return out;
  }
  if (value.getDefiningOp<StreamCreateOp>())
    return "stream.create";
  return "stream";
}

static void printEscaped(raw_ostream &os, StringRef value) {
  for (char c : value) {
    if (c == '"' || c == '\\')
      os << '\\';
    if (c == '\n')
      os << "\\n";
    else
      os << c;
  }
}

static void appendDenseI32IfExists(SmallVectorImpl<int32_t> &out,
                                   DenseI32ArrayAttr attr) {
  if (attr)
    llvm::append_range(out, attr.asArrayRef());
}

unsigned TopologyGraph::addNode(InvokeOp invoke, KernelOp callee) {
  ProcessNode node;
  node.id = nodes.size();
  node.invoke = invoke;
  node.callee = callee;
  node.parent = invoke->getParentOfType<KernelOp>();
  assert(node.parent == scope && "invoke must belong to the graph scope");
  llvm::append_range(node.mapping, callee.getMapping());
  appendDenseI32IfExists(
      node.coord, callee->getAttrOfType<DenseI32ArrayAttr>(kCoordAttrName));
  appendDenseI32IfExists(
      node.grid, callee->getAttrOfType<DenseI32ArrayAttr>(kGridAttrName));
  assert(node.coord.size() == node.grid.size() &&
         "coord and grid must have the same size");
  node.isConcrete = !node.coord.empty();
  nodes.push_back(std::move(node));
  return nodes.back().id;
}

Channel &TopologyGraph::getOrAddChannel(Value stream, ArrayRef<int64_t> lane) {
  auto *it = llvm::find_if(
      channels, [&](Channel &channel) { return channel.isSame(stream, lane); });
  if (it != channels.end())
    return *it;

  Channel channel;
  channel.stream = stream;
  channel.streamType = cast<StreamType>(stream.getType());
  llvm::append_range(channel.lane, lane);
  channels.push_back(std::move(channel));
  return channels.back();
}

LogicalResult TopologyGraph::addEndpoint(unsigned nodeId, InvokeOp invoke,
                                         Operation *streamOp, Value stream,
                                         Endpoint::Kind kind,
                                         bool skipDynamicLanes) {
  // local stream is not visible to the caller
  auto arg = dyn_cast<BlockArgument>(stream);
  if (!arg)
    return success();
  auto callee = nodes[nodeId].callee;
  if (arg.getOwner()->getParentOp() != callee)
    return success();

  unsigned argNo = arg.getArgNumber();
  assert(argNo < callee.getNumArguments() &&
         "stream argument must be a valid argument of the callee");

  Value callerStream = invoke->getOperand(argNo);

  ValueRange indices;
  if (auto get = dyn_cast<StreamGetOp>(streamOp))
    indices = get.getIndices();
  else
    indices = cast<StreamPutOp>(streamOp).getIndices();

  auto laneOr = getStaticLane(indices);
  if (failed(laneOr)) {
    if (skipDynamicLanes)
      return success();
    return streamOp->emitError("stream lane index must be static");
  }

  Channel &channel = getOrAddChannel(callerStream, *laneOr);
  Endpoint endpoint;
  endpoint.nodeId = nodeId;
  endpoint.argNo = argNo;
  endpoint.streamOp = streamOp;
  endpoint.kind = kind;
  llvm::append_range(endpoint.lane, *laneOr);

  if (kind == Endpoint::Kind::Producer)
    channel.producers.push_back(std::move(endpoint));
  else
    channel.consumers.push_back(std::move(endpoint));
  return success();
}

FailureOr<TopologyGraph>
allo::buildTopologyGraph(KernelOp scope, SymbolTableCollection &symbols,
                         bool skipDynamicLanes) {
  TopologyGraph graph(scope);
  if (scope.getBody().empty())
    return scope.emitError("cannot build topology graph for external kernel");

  SmallVector<InvokeOp> invokes;
  scope.walk([&](InvokeOp invoke) { invokes.push_back(invoke); });

  for (InvokeOp invoke : invokes) {
    auto callee = symbols.lookupNearestSymbolFrom<KernelOp>(
        invoke, invoke.getCalleeAttr());
    assert(callee && "callee must be resolvable");

    unsigned nodeId = graph.addNode(invoke, callee);
    WalkResult walkResult = callee.walk([&](Operation *op) {
      if (auto get = dyn_cast<StreamGetOp>(op)) {
        if (failed(graph.addEndpoint(nodeId, invoke, op, get.getStream(),
                                     Endpoint::Kind::Consumer,
                                     skipDynamicLanes)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }

      if (auto put = dyn_cast<StreamPutOp>(op)) {
        if (failed(graph.addEndpoint(nodeId, invoke, op, put.getStream(),
                                     Endpoint::Kind::Producer,
                                     skipDynamicLanes)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
  }

  return graph;
}

void TopologyGraph::exportAsDot(raw_ostream &os) const {
  os << "digraph TopologyGraph {\n";
  os << "  rankdir=LR;\n";

  for (auto &node : nodes) {
    os << "  n" << node.id << " [shape=box,label=\"";
    os << "n" << node.id << ": @";
    printEscaped(os, const_cast<KernelOp &>(node.callee).getSymName());
    os << "\\nmapping=";
    printI32Array(os, node.mapping);
    if (!node.coord.empty()) {
      os << "\\ncoord=";
      printI32Array(os, node.coord);
    }
    os << "\"];\n";
  }

  for (auto indexed : llvm::enumerate(channels)) {
    auto &channel = indexed.value();
    os << "  c" << indexed.index() << " [shape=diamond,label=\"";
    os << "ch" << indexed.index() << "\\n";
    printEscaped(os, stringifyValue(channel.stream));
    printI64Array(os, channel.lane);
    os << "\\n";
    printEscaped(os, stringifyType(channel.streamType));
    os << "\"];\n";

    for (auto &producer : channel.producers)
      os << "  n" << producer.nodeId << " -> c" << indexed.index()
         << " [label=\"put\"];\n";
    for (auto &consumer : channel.consumers)
      os << "  c" << indexed.index() << " -> n" << consumer.nodeId
         << " [label=\"get\"];\n";
  }

  os << "}\n";
}
