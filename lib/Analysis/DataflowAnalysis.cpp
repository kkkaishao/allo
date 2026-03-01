#include "allo/Analysis/DataflowAnalysis.h"

#include "allo/TransformOps/Utils.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::allo;

static bool belongsToKernel(Operation *op, KernelOp kernel) {
  return op && op->getParentOfType<KernelOp>() == kernel;
}

static std::string affineMapToString(AffineMap map) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << map;
  return out;
}

static FailureOr<std::string>
stringifyValueForSignature(Value value, ArrayRef<Value> inductionVars,
                           DenseMap<Value, std::string> &memo);

static FailureOr<std::string>
stringifyAffineApplyForSignature(affine::AffineApplyOp applyOp,
                                 ArrayRef<Value> inductionVars,
                                 DenseMap<Value, std::string> &memo) {
  SmallVector<std::string, 4> operandSigs;
  operandSigs.reserve(applyOp.getOperands().size());
  for (Value operand : applyOp.getOperands()) {
    auto sig = stringifyValueForSignature(operand, inductionVars, memo);
    if (failed(sig))
      return failure();
    operandSigs.push_back(*sig);
  }

  std::string out = "A(" + affineMapToString(applyOp.getAffineMap()) + "|";
  for (unsigned i = 0; i < operandSigs.size(); ++i) {
    out += operandSigs[i];
    if (i + 1 != operandSigs.size())
      out += ",";
  }
  out += ")";
  return out;
}

static FailureOr<std::string>
stringifyValueForSignature(Value value, ArrayRef<Value> inductionVars,
                           DenseMap<Value, std::string> &memo) {
  if (auto it = memo.find(value); it != memo.end())
    return it->second;
  value = stripCast(value);
  // Case 1: value is an induction var
  if (auto *ivIt = llvm::find(inductionVars, value);
      ivIt != inductionVars.end()) {
    std::string sig =
        "IV" + std::to_string(std::distance(inductionVars.begin(), ivIt));
    memo[value] = sig;
    return sig;
  }
  // Case 2: constant value
  IntegerAttr::ValueType cst;
  if (matchPattern(value, m_ConstantInt(&cst))) {
    std::string sig = "C" + std::to_string(cst.getSExtValue());
    memo[value] = sig;
    return sig;
  }
  // Case 3: func/kernel arguments
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *owner = blockArg.getOwner();
    if (owner->getParentOp()->getParentOfType<KernelOp>()) {
      std::string sig = "ARG" + std::to_string(blockArg.getArgNumber());
      memo[value] = sig;
      return sig;
    }
    return failure();
  }
  // Case 4: affine.apply result
  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return failure();
  if (auto applyOp = dyn_cast<affine::AffineApplyOp>(defOp)) {
    auto sig = stringifyAffineApplyForSignature(applyOp, inductionVars, memo);
    if (failed(sig))
      return failure();
    memo[value] = *sig;
    return sig;
  }
  return failure();
}

static FailureOr<std::string> stringifyOperands(ValueRange operands,
                                                ArrayRef<Value> inductionVars) {
  DenseMap<Value, std::string> memo;
  std::string out;
  for (size_t i = 0; i < operands.size(); ++i) {
    auto sig = stringifyValueForSignature(operands[i], inductionVars, memo);
    if (failed(sig))
      return failure();
    out += *sig;
    if (i + 1 != operands.size())
      out += ",";
  }
  return out;
}

static FailureOr<std::string>
stringifyAffineForSignature(affine::AffineForOp forOp,
                            ArrayRef<Value> inductionVars) {
  DenseMap<Value, std::string> memo;
  std::string out = "for(";
  out += affineMapToString(forOp.getLowerBoundMap());
  out += "|";
  for (size_t i = 0; i < forOp.getLowerBoundOperands().size(); ++i) {
    auto sig = stringifyValueForSignature(forOp.getLowerBoundOperands()[i],
                                          inductionVars, memo);
    if (!failed(sig))
      return failure();
    out += *sig;
    if (i + 1 != forOp.getLowerBoundOperands().size())
      out += ",";
  }
  out += ";";
  out += affineMapToString(forOp.getUpperBoundMap());
  out += "|";
  for (size_t i = 0; i < forOp.getUpperBoundOperands().size(); ++i) {
    auto sig = stringifyValueForSignature(forOp.getUpperBoundOperands()[i],
                                          inductionVars, memo);
    if (failed(sig))
      return failure();
    out += *sig;
    if (i + 1 != forOp.getUpperBoundOperands().size())
      out += ",";
  }
  out += ";s" + std::to_string(forOp.getStep().getSExtValue()) + ")";
  return out;
}

static FailureOr<std::string>
stringifyAffineParallelSignature(affine::AffineParallelOp parOp,
                                 ArrayRef<Value> inductionVars) {
  DenseMap<Value, std::string> memo;
  std::string out = "par(";
  auto lbMap = parOp.getLowerBoundsMap();
  auto ubMap = parOp.getUpperBoundsMap();
  auto steps = parOp.getSteps();

  out += affineMapToString(lbMap);
  out += "|";
  for (unsigned j = 0; j < parOp.getLowerBoundsOperands().size(); ++j) {
    auto sig = stringifyValueForSignature(parOp.getLowerBoundsOperands()[j],
                                          inductionVars, memo);
    if (failed(sig))
      return failure();
    out += *sig;
    if (j + 1 != parOp.getLowerBoundsOperands().size())
      out += ",";
  }
  out += ";";
  out += affineMapToString(ubMap);
  out += "|";
  for (unsigned j = 0; j < parOp.getUpperBoundsOperands().size(); ++j) {
    auto sig = stringifyValueForSignature(parOp.getUpperBoundsOperands()[j],
                                          inductionVars, memo);
    if (failed(sig))
      return failure();
    out += *sig;
    if (j + 1 != parOp.getUpperBoundsOperands().size())
      out += ",";
  }
  out += ";steps=";
  for (size_t i = 0; i < steps.size(); ++i) {
    out += std::to_string(steps[i]);
    if (i + 1 != steps.size())
      out += ",";
  }
  out += ")";
  return out;
}

static FailureOr<std::string> stringifyLoopNest(Operation *op,
                                                SmallVectorImpl<Value> &ivs) {
  SmallVector<Operation *, 8> affineOps;
  affine::getEnclosingAffineOps(*op, &affineOps);
  // collect ivs in order from outermost to innermost
  for (Operation *affineOp : affineOps) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(affineOp)) {
      ivs.push_back(forOp.getInductionVar());
    } else if (auto parOp = dyn_cast<affine::AffineParallelOp>(affineOp)) {
      ivs.append(parOp.getIVs().begin(), parOp.getIVs().end());
    } else {
      return failure();
    }
  }

  std::string out;
  for (unsigned i = 0; i < affineOps.size(); ++i) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(affineOps[i])) {
      auto sig = stringifyAffineForSignature(forOp, ivs);
      if (failed(sig))
        return failure();
      out += *sig;
    } else if (auto parOp = dyn_cast<affine::AffineParallelOp>(affineOps[i])) {
      auto sig = stringifyAffineParallelSignature(parOp, ivs);
      if (failed(sig))
        return failure();
      out += *sig;
    }
    if (i + 1 != affineOps.size())
      out += "->";
  }
  if (out.empty())
    out = "none";
  return out;
}

// Generate a signature string for the access site, which includes the loop nest
// structure and the given operands.
static FailureOr<std::string> getAccessSiteSignature(Operation *op,
                                                     ValueRange operands) {
  SmallVector<Value, 8> inductionVars;
  auto loopSig = stringifyLoopNest(op, inductionVars);
  if (failed(loopSig))
    return failure();

  auto idxSig = stringifyOperands(operands, inductionVars);
  if (failed(idxSig))
    return failure();
  return *loopSig + "::" + *idxSig;
}

StringAttr DataflowGraph::getChannelAttr(StringRef channel) const {
  return StringAttr::get(Ctx, channel);
}

ChannelAccessKind
DataflowGraph::getChannelAccessKind(ChannelAccessOpInterface op) {
  if (isa<ChanPutOp>(op))
    return ChannelAccessKind::Put;
  if (isa<ChanGetOp>(op))
    return ChannelAccessKind::Get;
  if (isa<ChanAcquireOp>(op))
    return ChannelAccessKind::Acquire;
  return ChannelAccessKind::Release;
}

bool DataflowGraph::isPutGetOp(Operation *op) {
  return isa<ChanPutOp, ChanGetOp>(op);
}

bool DataflowGraph::isChannelWriteOp(ChannelAccessOpInterface op) {
  return isa<ChanPutOp, ChanReleaseOp>(op);
}

bool DataflowGraph::isChannelReadOp(ChannelAccessOpInterface op) {
  return isa<ChanGetOp, ChanAcquireOp>(op);
}

// return the node for the kernel, create one if it doesn't exist;
DataflowNode *DataflowGraph::addNode(KernelOp kernel) {
  assert(kernel && "cannot add null kernel to dataflow graph");
  auto it = NodeMap.find(kernel);
  if (it != NodeMap.end()) {
    assert(it->second && "dataflow graph node map contains null pointer");
    return it->second;
  }

  auto node = std::make_unique<DataflowNode>(kernel);
  DataflowNode *nodePtr = node.get();
  Nodes.push_back(std::move(node));
  NodeMap[kernel] = nodePtr;
  // initialize in/out edges vec
  InEdges[nodePtr];
  OutEdges[nodePtr];
  return nodePtr;
}

void DataflowGraph::init() {
  for (auto kernel : Mod.getOps<KernelOp>())
    addNode(kernel);

  // collect channel accesses and store access points
  Mod.walk([&](ChannelAccessOpInterface chanOp) {
    StringAttr chanAttr = chanOp.getChannelAttr().getAttr();
    auto &accessPoints = ChannelAccesses[chanAttr];
    if (isChannelReadOp(chanOp))
      accessPoints.ReadOps.insert(chanOp);
    else if (isChannelWriteOp(chanOp))
      accessPoints.WriteOps.insert(chanOp);
  });

  // for each channel, create edges between all combinations of write and read
  // access points
  for (auto &[chan, points] : ChannelAccesses) {
    for (auto writeOp : points.WriteOps) {
      for (auto readOp : points.ReadOps) {
        addEdge(chan, writeOp, readOp);
      }
    }
  }
}

LogicalResult DataflowGraph::removeEdge(DataflowEdge *edge) {
  if (!edge)
    return failure();

  // delete from in/out edges
  auto outIt = OutEdges.find(edge->SrcNode);
  if (outIt != OutEdges.end())
    llvm::erase_if(outIt->second, [&](DataflowEdge *e) { return e == edge; });

  auto inIt = InEdges.find(edge->DstNode);
  if (inIt != InEdges.end())
    llvm::erase_if(inIt->second, [&](DataflowEdge *e) { return e == edge; });

  // erase from main edge list
  auto *edgeIt = llvm::find_if(
      Edges, [&](const auto &ownedEdge) { return ownedEdge.get() == edge; });
  if (edgeIt == Edges.end())
    return failure();
  // erase entry
  Edges.erase(edgeIt);
  return success();
}

LogicalResult DataflowGraph::removeNode(DataflowNode *node) {
  if (!node)
    return failure();
  auto nodeIt = NodeMap.find(node->Kernel);
  if (nodeIt == NodeMap.end())
    return failure();
  // remove all the in/out edges of the node
  llvm::SmallDenseSet<DataflowEdge *, 8> edgesToRemove;
  auto inIt = InEdges.find(node);
  if (inIt != InEdges.end()) {
    edgesToRemove.insert_range(inIt->second);
  }
  auto outIt = OutEdges.find(node);
  if (outIt != OutEdges.end()) {
    edgesToRemove.insert_range(outIt->second);
  }
  for (auto *edge : edgesToRemove) {
    (void)removeEdge(edge);
  }
  // delete the entry for the node
  InEdges.erase(node);
  OutEdges.erase(node);
  NodeMap.erase(nodeIt);
  llvm::erase_if(Nodes, [&](const auto &n) { return n.get() == node; });

  // delete all the access points belonging to the kernel
  for (auto it = ChannelAccesses.begin(); it != ChannelAccesses.end();) {
    auto current = it++;
    auto &accessPoints = current->second;
    accessPoints.WriteOps.remove_if(
        [&](auto op) { return belongsToKernel(op, node->Kernel); });
    accessPoints.ReadOps.remove_if(
        [&](auto op) { return belongsToKernel(op, node->Kernel); });
  }
  return success();
}

DataflowEdge *DataflowGraph::addEdge(StringAttr channel,
                                     ChannelAccessOpInterface srcOp,
                                     ChannelAccessOpInterface dstOp) {
  // precondition checks before adding the edge
  // check 1:
  if (!isChannelWriteOp(srcOp) || !isChannelReadOp(dstOp))
    return nullptr;
  // check 2:
  StringAttr srcChan = srcOp.getChannelAttr().getAttr();
  StringAttr dstChan = dstOp.getChannelAttr().getAttr();
  if (!srcChan || !dstChan || srcChan != channel || dstChan != channel)
    return nullptr;

  // add or get src/dst kernels
  auto srcKernel = srcOp->getParentOfType<KernelOp>();
  auto dstKernel = dstOp->getParentOfType<KernelOp>();
  if (!srcKernel || !dstKernel)
    return nullptr;

  DataflowNode *srcNode = addNode(srcKernel);
  DataflowNode *dstNode = addNode(dstKernel);

  // make edge
  auto edge = std::make_unique<DataflowEdge>();
  edge->SrcNode = srcNode;
  edge->DstNode = dstNode;
  edge->SrcOp = srcOp;
  edge->DstOp = dstOp;
  edge->Channel = channel;

  // update edge info
  DataflowEdge *edgePtr = edge.get();
  InEdges[dstNode].push_back(edgePtr);
  OutEdges[srcNode].push_back(edgePtr);
  Edges.push_back(std::move(edge));

  // update channel access points
  auto &accessPoints = ChannelAccesses[channel];
  accessPoints.WriteOps.insert(srcOp);
  accessPoints.ReadOps.insert(dstOp);

  return edgePtr;
}

// Get edges from first to second, return empty if either kernel is null or no
// edge exists
DataflowGraph::DataflowEdgeList DataflowGraph::getEdge(KernelOp src,
                                                       KernelOp dst) const {
  auto srcIt = NodeMap.find(src);
  if (srcIt == NodeMap.end()) {
    return {};
  }
  auto dstIt = NodeMap.find(dst);
  if (dstIt == NodeMap.end()) {
    return {};
  }
  auto firstOuts = OutEdges.find(srcIt->second);
  DataflowEdgeList result;
  for (auto *edge : firstOuts->second) {
    if (edge->DstNode == dstIt->second) {
      result.push_back(edge);
    }
  }
  return result;
}

// return if there is any dependence from src to dst, regardless of channel
bool DataflowGraph::hasForwardDependence(KernelOp src, KernelOp dst) const {
  if (src == dst)
    return false;
  auto srcIt = NodeMap.find(src);
  if (srcIt == NodeMap.end())
    return false;
  auto dstIt = NodeMap.find(dst);
  if (dstIt == NodeMap.end())
    return false;

  auto srcOuts = OutEdges.find(srcIt->second);
  if (srcOuts == OutEdges.end())
    return false;
  if (llvm::any_of(srcOuts->second, [&](DataflowEdge *edge) {
        return edge->DstNode == dstIt->second;
      }))
    return true;
  return false;
}

// return if there is any dependence from src to dst through the given channel
bool DataflowGraph::hasForwardDependence(KernelOp src, KernelOp dst,
                                         StringRef channel) const {
  if (src == dst)
    return false;
  auto srcIt = NodeMap.find(src);
  if (srcIt == NodeMap.end())
    return false;
  auto dstIt = NodeMap.find(dst);
  if (dstIt == NodeMap.end())
    return false;

  auto srcOuts = OutEdges.find(srcIt->second);
  if (srcOuts == OutEdges.end())
    return false;
  if (llvm::any_of(srcOuts->second, [&](DataflowEdge *edge) {
        return edge->DstNode == dstIt->second &&
               edge->Channel.getValue() == channel;
      }))
    return true;
  return false;
}

// backward dependence is just the reverse of forward dependence
bool DataflowGraph::hasBackwardDependence(KernelOp src, KernelOp dst) const {
  return hasForwardDependence(dst, src);
}

// backward dependence is just the reverse of forward dependence
bool DataflowGraph::hasBackwardDependence(KernelOp src, KernelOp dst,
                                          StringRef channel) const {
  return hasForwardDependence(dst, src, channel);
}

bool DataflowGraph::hasDependence(KernelOp lhs, KernelOp rhs) const {
  return hasForwardDependence(lhs, rhs) || hasBackwardDependence(lhs, rhs);
}

// return producers kernels and consumer kernels for the given channel,
// and the communication pattern (SPSC, SPMC, MPSC, MPMC)
ChannelPatternInfo
DataflowGraph::analyzeChannelPattern(StringAttr channel) const {
  ChannelPatternInfo info;
  auto accessIt = ChannelAccesses.find(channel);
  if (accessIt == ChannelAccesses.end())
    return info;

  auto &writeOps = accessIt->second.WriteOps;
  auto &readOps = accessIt->second.ReadOps;
  info.WriteOps = accessIt->second.WriteOps.getArrayRef();
  info.ReadOps = accessIt->second.ReadOps.getArrayRef();

  if (readOps.size() == 1 && writeOps.size() == 1)
    info.Kind = ChannelPatternKind::SPSC;
  else if (readOps.size() > 1 && writeOps.size() == 1)
    info.Kind = ChannelPatternKind::SPMC;
  else if (readOps.size() == 1 && writeOps.size() > 1)
    info.Kind = ChannelPatternKind::MPSC;
  else if (readOps.size() > 1 && writeOps.size() > 1)
    info.Kind = ChannelPatternKind::MPMC;
  return info;
}

// Get a signature for the access site, including the surrounding control
// flow structure and the affine expressions for indices.
// If the access site cannot be analyzed using affine, return std::nullopt.
FailureOr<std::string>
DataflowGraph::getAccessSiteSignature(ChannelAccessOpInterface op) {
  return ::getAccessSiteSignature(op, op.getIndices());
}

FailureOr<SmallVector<ChannelForwardPair, 4>>
DataflowGraph::analyzePutGetForward(KernelOp producer, KernelOp consumer,
                                    StringAttr channel, bool emitError) const {
  InFlightDiagnostic diag = emitRemark(producer.getLoc());
  auto accessIt = ChannelAccesses.find(channel);
  if (accessIt == ChannelAccesses.end()) {
    if (emitError) {
      diag << "No access points found for channel " << channel.getValue()
           << "; cannot forward";
      return diag;
    }
    return failure();
  }
  // get all the write and read access points for the channel
  auto &writes = accessIt->second.WriteOps;
  auto &reads = accessIt->second.ReadOps;
  // fast checks:
  if (writes.size() != reads.size()) {
    if (emitError) {
      diag << "Producer and consumer have mismatched number of access points "
           << "for channel " << channel.getValue() << ": " << writes.size()
           << " write(s) vs " << reads.size() << " read(s); cannot forward";
      diag.attachNote(consumer.getLoc()) << "see consumer kernel here";
      return diag;
    }
    return failure();
  }
  if (writes.empty() || reads.empty()) {
    if (emitError) {
      diag << "Producer and consumer must have at least one access point for "
           << "channel " << channel.getValue() << " for forwarding";
      diag.attachNote(consumer.getLoc()) << "see consumer kernel here";
      return diag;
    }
    return failure();
  }
  // check if all access points belong to the correct kernels; if not,
  // forwarding may cause potential deadlock
  if (llvm::any_of(writes, [&](ChannelAccessOpInterface op) {
        auto parent = op->getParentOfType<KernelOp>();
        return !parent || parent != producer;
      })) {
    if (emitError) {
      diag << "Some write access points do not belong to producer kernel @"
           << producer.getSymName();
      diag.attachNote(producer.getLoc()) << "see producer kernel here";
      return diag;
    }
    return failure();
  }
  if (llvm::any_of(reads, [&](ChannelAccessOpInterface op) {
        auto parent = op->getParentOfType<KernelOp>();
        return !parent || parent != consumer;
      })) {
    if (emitError) {
      diag << "Some read access points do not belong to consumer kernel @"
           << consumer.getSymName();
      diag.attachNote(consumer.getLoc()) << "see consumer kernel here";
      return diag;
    }
    return failure();
  }

  llvm::StringMap<ChannelAccessOpInterface> writeBySig;
  llvm::StringMap<ChannelAccessOpInterface> readBySig;
  // generate signatures for all the access sites
  for (auto write : writes) {
    auto sig = getAccessSiteSignature(write);
    if (failed(sig)) {
      if (emitError) {
        diag << "Non-affine channel indices or loop bounds detected at write "
             << "access site in producer kernel @" << producer.getSymName()
             << "; run raise-to-affine first";
        diag.attachNote(write.getLoc()) << "see non-affine access here";
        return diag;
      }
      return failure();
    }
    if (!writeBySig.insert({*sig, write}).second) {
      if (emitError) {
        diag << "Duplicate access site signature detected for write access at "
             << "producer kernel @" << producer.getSymName()
             << "; cannot forward";
        auto existingWrite = writeBySig.lookup(*sig);
        diag.attachNote(existingWrite.getLoc())
            << "previous write access site with the same signature";
        diag.attachNote(write.getLoc())
            << "current write access site with the same signature";
        return diag;
      }
      return failure();
    }
  }
  for (auto read : reads) {
    auto sig = getAccessSiteSignature(read);
    if (failed(sig)) {
      if (emitError) {
        diag << "Non-affine channel indices or loop bounds detected at read "
             << "access site in consumer kernel @" << consumer.getSymName()
             << "; run raise-to-affine first";
        diag.attachNote(read.getLoc()) << "see non-affine access here";
        return diag;
      }
      return failure();
    }
    if (!readBySig.insert({*sig, read}).second) {
      if (emitError) {
        diag << "Duplicate access site signature detected for read access at "
             << "consumer kernel @" << consumer.getSymName()
             << "; cannot forward";
        auto existingRead = readBySig.lookup(*sig);
        diag.attachNote(existingRead.getLoc())
            << "previous read access site with the same signature";
        diag.attachNote(read.getLoc())
            << "current read access site with the same signature";
        return diag;
      }
      return failure();
    }
  }

  // check if all write access can have matching read access with the same
  // signature; if not, forwarding will cause potential deadlock due to
  // mismatched number of sends and receives
  for (auto &[sig, _] : writeBySig) {
    auto readIt = readBySig.find(sig);
    if (readIt == readBySig.end()) {
      if (emitError) {
        diag << "No matching read access site found for write access site"
             << " at producer kernel @" << producer.getSymName()
             << "; cannot forward";
        diag.attachNote(writeBySig.lookup(sig).getLoc())
            << "see write access site here";
        return diag;
      }
      return failure();
    }
    // require get->put or acquire-release pair for forwarding (semantic check)
    if (!(isa<ChanAcquireOp>(readIt->second) &&
          isa<ChanReleaseOp>(writeBySig[sig])) &&
        !(isa<ChanGetOp>(readIt->second) && isa<ChanPutOp>(writeBySig[sig]))) {
      if (emitError) {
        diag << "Mismatched access types for signature " << sig
             << ": write access is "
             << writeBySig[sig]->getName().getStringRef()
             << " but read access is "
             << readIt->second->getName().getStringRef()
             << "; cannot forward between put/get and acquire/release";
        diag.attachNote(writeBySig.lookup(sig).getLoc())
            << "see write access site here";
        diag.attachNote(readIt->second.getLoc()) << "see read access site here";
        return diag;
      }
      return failure();
    }
  }

  // fill up the matched pairs for potential forwarding
  SmallVector<ChannelForwardPair, 4> forwardPairs;
  for (auto &[sig, write] : writeBySig) {
    auto read = readBySig.lookup(sig);
    forwardPairs.push_back({write, read});
  }
  diag.abandon();
  return forwardPairs;
}

bool DataflowGraph::hasCycle() const {
  DenseMap<KernelOp, unsigned> inDegree;
  DenseMap<KernelOp, SmallVector<KernelOp, 4>> successors;
  inDegree.reserve(NodeMap.size());
  successors.reserve(NodeMap.size());
  // initialize in-degree and successor map
  for (const auto &[k, _] : NodeMap) {
    inDegree[k] = 0;
    successors[k];
  }
  // populate in-degree and successor map based on edges
  for (const auto &edge : Edges) {
    KernelOp src = edge->SrcNode->Kernel;
    KernelOp dst = edge->DstNode->Kernel;
    // ignore self loops if any, as they do not contribute to cycles in the
    // graph
    if (src == dst)
      continue;
    successors[src].push_back(dst);
    ++inDegree[dst];
  }
  // initialize worklist with source nodes (in-degree 0)
  SmallVector<KernelOp, 8> worklist;
  worklist.reserve(inDegree.size());
  for (const auto &[k, degree] : inDegree) {
    if (degree == 0)
      worklist.push_back(k);
  }
  // perform Kahn's algorithm for topological sorting;
  // if we cannot visit all nodes, there is a cycle in the graph
  size_t visited = 0;
  while (!worklist.empty()) {
    KernelOp current = worklist.pop_back_val();
    ++visited;

    auto outIt = successors.find(current);
    if (outIt == successors.end() || outIt->second.empty())
      continue;

    for (KernelOp dst : outIt->second) {
      auto degreeIt = inDegree.find(dst);
      if (degreeIt == inDegree.end() || degreeIt->second == 0)
        continue;

      --degreeIt->second;
      if (degreeIt->second == 0)
        worklist.push_back(dst);
    }
  }

  return visited != inDegree.size();
}

LogicalResult DataflowGraph::mergeNodes(KernelOp kernelA, KernelOp kernelB,
                                        KernelOp mergedKernel,
                                        DenseSet<StringAttr> &ignores) {
  if (kernelA == kernelB)
    return failure();

  auto nodeAIt = NodeMap.find(kernelA);
  auto nodeBIt = NodeMap.find(kernelB);
  if (nodeAIt == NodeMap.end() || nodeBIt == NodeMap.end())
    return failure();

  DataflowNode *nodeA = nodeAIt->second;
  DataflowNode *nodeB = nodeBIt->second;

  DataflowNode *mergedNode = addNode(mergedKernel);
  if (!mergedNode)
    return failure();

  // Collect all incident edges first since we will mutate edge lists.
  llvm::SmallSetVector<DataflowEdge *, 4> incidentEdgeSet;
  auto collectEdges = [&](DataflowNode *node) {
    if (auto inIt = InEdges.find(node); inIt != InEdges.end())
      incidentEdgeSet.insert_range(inIt->second);
    if (auto outIt = OutEdges.find(node); outIt != OutEdges.end())
      incidentEdgeSet.insert_range(outIt->second);
  };
  collectEdges(nodeA);
  collectEdges(nodeB);
  auto incidentEdges = incidentEdgeSet.takeVector();

  for (DataflowEdge *edge : incidentEdges) {
    DataflowNode *oldSrc = edge->SrcNode;
    DataflowNode *oldDst = edge->DstNode;
    DataflowNode *newSrc =
        (oldSrc == nodeA || oldSrc == nodeB) ? mergedNode : oldSrc;
    DataflowNode *newDst =
        (oldDst == nodeA || oldDst == nodeB) ? mergedNode : oldDst;

    // Internal dependences are removed by default after merge.
    // But channels in `ignores` must be preserved even if they form self loops.
    if (newSrc == mergedNode && newDst == mergedNode &&
        !ignores.contains(edge->Channel)) {
      (void)removeEdge(edge);
      continue;
    }
    if (newSrc == mergedNode) {
      edge->SrcNode = newSrc;
      OutEdges[mergedNode].push_back(edge);
    }
    if (newDst == mergedNode) {
      edge->DstNode = newDst;
      InEdges[mergedNode].push_back(edge);
    }
  }

  (void)removeNode(nodeA);
  (void)removeNode(nodeB);

  // update channel access points
  mergedKernel.walk([&](ChannelAccessOpInterface op) {
    auto chanAttr = op.getChannelAttr().getAttr();
    if (isChannelReadOp(op))
      ChannelAccesses[chanAttr].ReadOps.insert(op);
    else if (isChannelWriteOp(op))
      ChannelAccesses[chanAttr].WriteOps.insert(op);
  });

  return success();
}
