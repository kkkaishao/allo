/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "allo/Translation/VivadoHLSEmitter.h"

#include <cctype>

using namespace mlir;
using namespace mlir::allo;

static std::string getIntegerTypeName(unsigned width, bool isSigned) {
  std::string prefix = isSigned ? "" : "u";
  switch (width) {
  case 1:
    return "bool";
  case 8:
  case 16:
  case 32:
  case 64:
    return prefix + "int" + std::to_string(width) + "_t";
  default:
    return (isSigned ? "ap_int<" : "ap_uint<") + std::to_string(width) + ">";
  }
}

// The "allo.signed" marker carries one char per function operand then result:
// 's' signed integer, 'u' unsigned integer, 'x' non-integer. A missing or short
// marker falls back to unsigned, preserving prior behavior.
static bool operandIsSigned(func::FuncOp func, unsigned idx) {
  auto attr = func->getAttrOfType<StringAttr>("allo.signed");
  if (!attr)
    return false;
  StringRef marker = attr.getValue();
  return idx < marker.size() && marker[idx] == 's';
}

// A C++ literal for the low ``width`` set bits, e.g. ``0xfULL`` for width 4.
// Used to mask out a bit slice; only widths up to 64 fit in a single literal.
static std::string getBitMaskLiteral(unsigned width) {
  assert(width >= 1 && width <= 64 && "bit slice width out of range");
  uint64_t mask = width == 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
  return "0x" + llvm::utohexstr(mask, /*LowerCase=*/true) + "ULL";
}

static std::string sanitizeCppIdentifier(llvm::StringRef name) {
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    unsigned char uc = static_cast<unsigned char>(c);
    result.push_back(std::isalnum(uc) || c == '_' ? c : '_');
  }
  if (result.empty() ||
      std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

std::string VivadoHLSEmitter::getSymbolName(llvm::StringRef name) {
  auto existing = symbolNameTable.find(name);
  if (existing != symbolNameTable.end())
    return existing->second;

  std::string base = sanitizeCppIdentifier(name);
  std::string unique = base;
  unsigned suffix = 0;
  while (usedSymbolNames.contains(unique))
    unique = base + "_" + std::to_string(++suffix);

  usedSymbolNames.insert(unique);
  symbolNameTable[name] = unique;
  return unique;
}

std::string VivadoHLSEmitter::getTemporaryName(llvm::StringRef prefix) {
  return (prefix + std::to_string(temporaryNameCounter++)).str();
}

std::string VivadoHLSEmitter::getPrimitiveTypeName(Type type, bool isSigned) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    type = shapedType.getElementType();
  /// Primitive types
  if (isa<Float16Type>(type))
    return "half";
  if (isa<Float32Type>(type))
    return "float";
  if (isa<Float64Type>(type))
    return "double";

  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();
    return getIntegerTypeName(width, isSigned);
  }

  if (isa<IndexType>(type)) {
    unsigned width = state.indexWidth;
    bool isSigned = true; // index type is signed in MLIR
    return getIntegerTypeName(width, isSigned);
  }

  if (auto fixed = dyn_cast<FixedType>(type)) {
    unsigned width = fixed.getWidth();
    unsigned frac = fixed.getFrac();
    return "ap_fixed<" + std::to_string(width) + ", " +
           std::to_string(width - frac) + ">";
  }

  if (auto ufixed = dyn_cast<UFixedType>(type)) {
    unsigned width = ufixed.getWidth();
    unsigned frac = ufixed.getFrac();
    return "ap_ufixed<" + std::to_string(width) + ", " +
           std::to_string(width - frac) + ">";
  }

  emitError(UnknownLoc::get(type.getContext()))
      << "unsupported type in Vivado HLS emitter: " << type;
  state.failed = true;
  return "/*unsupported_type*/";
}

// A block payload (a shaped base type) is streamed element-by-element through a
// scalar FIFO of its element type, rather than via hls::stream_of_blocks. A
// plain FIFO synthesizes anywhere (stream_of_blocks is restricted to dataflow
// regions and forbids multiple put/get in the same region) and, since the block
// is materialized as a local array either way, costs no more hardware.
std::string VivadoHLSEmitter::getStreamTypeName(StreamType type) {
  return "hls::stream<" + getPrimitiveTypeName(type.getBaseType()) + ">";
}

// FIFO depth in elements: a block payload needs depth-many blocks buffered,
// each of which is `product(blockShape)` scalar elements.
static std::size_t streamFifoDepth(StreamType type) {
  std::size_t depth = type.getDepth();
  if (auto shaped = dyn_cast<ShapedType>(type.getBaseType()))
    for (int64_t dim : shaped.getShape())
      depth *= static_cast<std::size_t>(dim);
  return depth;
}

void VivadoHLSEmitter::emitFunction(func::FuncOp func) {
  if (func.getBlocks().empty())
    return;

  if (func.getBlocks().size() > 1) {
    func->emitError() << "Multiple blocks in a function are not supported in "
                         "Vivado HLS emitter.";
    state.failed = true;
    return;
  }

  emitFunctionSignature(func);
  state.os << " {\n";
  state.addIndent();
  // emit function-level directives
  emitFunctionDirectives(func);
  emitBlock(func.getBlocks().front());
  state.reduceIndent();
  state.os << "}\n";
}

void VivadoHLSEmitter::emitFunctionReturnType(func::FuncOp func) {
  unsigned nResults = func.getNumResults();
  if (nResults == 0)
    state.os << "void";
  else if (nResults == 1)
    state.os << getPrimitiveTypeName(
        func.getResultTypes().front(),
        operandIsSigned(func, func.getNumArguments()));
  else {
    func->emitError()
        << "Multiple return values are not supported in Vivado HLS emitter.";
    state.failed = true;
  }
}

void VivadoHLSEmitter::emitFunctionSignature(func::FuncOp func) {
  emitFunctionReturnType(func);
  state.os << " " << getSymbolName(func.getSymName()) << "(";
  emitFunctionArguments(func);
  state.os << ")";
}

void VivadoHLSEmitter::emitTrailingLocation(Operation *op) {
  if (state.withLocation) {
    if (auto loc = dyn_cast<FileLineColLoc>(op->getLoc()))
      state.os << "\t// " << loc.getFilename() << ":" << loc.getLine() << ":"
               << loc.getColumn();
  }
  state.os << "\n";
}

void VivadoHLSEmitter::emitFunctionArguments(func::FuncOp func) {
  for (auto arg : func.getArguments()) {
    if (arg != func.getArguments().front())
      state.os << ", ";
    if (auto streamType = dyn_cast<StreamType>(arg.getType())) {
      state.os << getStreamTypeName(streamType);
      if (streamType.getShape().empty())
        state.os << " &";
      else
        state.os << " ";
      state.os << state.getOrAddName(arg);
      emitArraySuffix(streamType.getShape(), arg.getLoc());
      continue;
    }
    state.os << getPrimitiveTypeName(arg.getType(),
                                     operandIsSigned(func, arg.getArgNumber()))
             << " " << state.getOrAddName(arg);
    if (auto shaped = dyn_cast<ShapedType>(arg.getType()))
      emitArraySuffix(shaped, arg.getLoc());
  }
}

void VivadoHLSEmitter::emitFunctionDirectives(func::FuncOp func) {
  if (func->hasAttr("dataflow")) {
    state.os.indent(state.currentIndent);
    state.os << "#pragma HLS dataflow\n";
  }
  state.os.indent(state.currentIndent);
  if (func->hasAttr("inline"))
    state.os << "#pragma HLS inline\n";
  else
    state.os << "#pragma HLS inline off\n";

  for (auto arg : func.getArguments()) {
    auto streamType = dyn_cast<StreamType>(arg.getType());
    if (!streamType)
      continue;
    state.os.indent(state.currentIndent);
    state.os << "#pragma HLS stream variable=" << state.getName(arg)
             << " depth=" << streamFifoDepth(streamType) << "\n";
  }

  auto argAttrs = func.getArgAttrs();
  if (!argAttrs) {
    state.os << "\n";
    return;
  }

  // emit partition directives for arguments
  for (auto [arg, attr] : llvm::zip(func.getArguments(), *argAttrs)) {
    auto dict = cast<DictionaryAttr>(attr);
    auto partOr = dict.getNamed("allo.part");
    if (!partOr)
      continue;
    auto partAttr = cast<allo::PartitionAttr>(partOr->getValue());
    emitPartitionAttr(partAttr, arg);
  }
  state.os << "\n";
}

void VivadoHLSEmitter::emitCall(func::CallOp op) {
  llvm::raw_ostream &os = state.os;
  // cpp cannot handle multiple return values
  if (op->getNumResults() > 1) {
    op->emitError()
        << "Multiple call results are not supported in Vivado HLS emitter.";
    state.failed = true;
    return;
  }
  if (op->getNumResults() == 1) {
    emitValueDecl(op.getResult(0));
    os << " = ";
  }
  os << getSymbolName(op.getCallee()) << "(";
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (i > 0)
      os << ", ";
    emitValueRef(op.getOperand(i));
  }
  os << ");";
}

void VivadoHLSEmitter::emitPartitionAttr(allo::PartitionAttr attr,
                                         Value value) {
  for (auto axiAttr : attr.getPartitions()) {
    state.os.indent(state.currentIndent);
    state.os << "#pragma HLS array_partition variable=" << state.getName(value);
    state.os << " dim=" << axiAttr.getDim();
    state.os << " factor=" << axiAttr.getFactor();
    switch (axiAttr.getKind()) {
    case allo::PartitionKindEnum::CyclicPartition:
      state.os << " cyclic";
      break;
    case allo::PartitionKindEnum::BlockPartition:
      state.os << " block";
      break;
    case allo::PartitionKindEnum::CompletePartition:
      state.os << " complete";
      break;
    }
    state.os << "\n";
  }
}

void VivadoHLSEmitter::emitAffineFor(affine::AffineForOp op) {
  llvm::raw_ostream &os = state.os;
  // declare variables for iter args
  for (auto [result, iter, init] :
       llvm::zip(op.getResults(), op.getRegionIterArgs(), op.getInits())) {
    emitValueDecl(iter);
    os << " = ";
    emitValueRef(init);
    os << ";\n";
    state.nameTable[result] = state.getName(iter);
  }

  if (op.getNumResults())
    os.indent(state.currentIndent);
  os << "for (";
  emitValueDecl(op.getInductionVar());
  os << " = ";
  std::string ivName = state.getName(op.getInductionVar());
  AffineMap lbMap = op.getLowerBoundMap();
  // if lb num results > 1, affine.for will take the max of all results as the
  // lower bound
  if (lbMap.getNumResults() > 1)
    emitAffineMapReduction(lbMap, op.getLowerBoundOperands(), "std::max");
  else
    AffineExprEmitter(state, op.getLowerBoundOperands(), lbMap.getNumDims())
        .emitAffineMap(lbMap);
  // if ub num results > 1, affine.for will take the min of all results as the
  // upper bound
  os << "; " << ivName << " < ";
  AffineMap ubMap = op.getUpperBoundMap();
  if (ubMap.getNumResults() > 1)
    emitAffineMapReduction(ubMap, op.getUpperBoundOperands(), "std::min");
  else
    AffineExprEmitter(state, op.getUpperBoundOperands(), ubMap.getNumDims())
        .emitAffineMap(ubMap);
  // emit step
  os << "; " << ivName << " += " << op.getStep() << ") {\n";
  state.addIndent();
  // emit pragmas
  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  state.reduceIndent();
  os.indent(state.currentIndent);
  os << "}";
}

void VivadoHLSEmitter::emitLoopDirectives(Operation *op) {
  if (auto unrollAttr = op->getAttrOfType<IntegerAttr>("unroll.f")) {
    int64_t unrollFactor = unrollAttr.getInt();
    state.os.indent(state.currentIndent);
    if (unrollFactor == 0)
      state.os << "#pragma HLS unroll\n";
    else
      state.os << "#pragma HLS unroll factor=" << unrollFactor << "\n";
  }
  if (auto pipelineAttr = op->getAttrOfType<IntegerAttr>("pipeline.ii")) {
    int64_t ii = pipelineAttr.getInt();
    state.os.indent(state.currentIndent);
    state.os << "#pragma HLS pipeline II=" << ii << "\n";
  }
}

void VivadoHLSEmitter::emitAffineLoad(affine::AffineLoadOp op) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op.getResult());
  os << " = ";
  AffineMap indexMap = op.getAffineMap();
  AffineExprEmitter indexEmitter(state, op.getMapOperands(),
                                 indexMap.getNumDims());
  emitValueRef(op.getMemref());
  for (unsigned i = 0; i < indexMap.getNumResults(); ++i) {
    os << "[";
    indexEmitter.visit(indexMap.getResult(i));
    os << "]";
  }
  os << ";";
}

void VivadoHLSEmitter::emitAffineStore(affine::AffineStoreOp op) {
  llvm::raw_ostream &os = state.os;
  AffineMap indexMap = op.getAffineMap();
  AffineExprEmitter indexEmitter(state, op.getMapOperands(),
                                 indexMap.getNumDims());
  emitValueRef(op.getMemref());
  for (unsigned i = 0; i < indexMap.getNumResults(); ++i) {
    os << "[";
    indexEmitter.visit(indexMap.getResult(i));
    os << "]";
  }
  os << " = ";
  emitValueRef(op.getValueToStore());
  os << ";";
}

void VivadoHLSEmitter::emitAffineIf(affine::AffineIfOp op) {
  llvm::raw_ostream &os = state.os;
  // emit results
  for (auto result : op.getResults()) {
    emitValueDecl(result);
    os << ";\n"; // leave it uninitialized for now, will be assigned in the
                 // then/else blocks
  }
  if (op.getNumResults())
    os.indent(state.currentIndent);
  os << "if (";

  IntegerSet conds = op.getCondition();
  AffineExprEmitter condEmitter(state, op->getOperands(), conds.getNumDims());
  unsigned nConds = conds.getNumConstraints();
  unsigned condIdx = 0;
  for (auto [cond, eq] :
       llvm::zip(conds.getConstraints(), conds.getEqFlags())) {
    condEmitter.visit(cond);
    if (eq) {
      os << " == 0";
    } else {
      os << " >= 0";
    }
    if (++condIdx != nConds)
      os << " && ";
  }
  os << ") {\n";
  state.addIndent();
  emitBlock(*op.getThenBlock());
  state.reduceIndent();
  os.indent(state.currentIndent);
  os << "}";

  if (op.hasElse()) {
    os << " else {\n";
    state.addIndent();
    emitBlock(*op.getElseBlock());
    state.reduceIndent();
    os.indent(state.currentIndent);
    os << "}";
  }
}

void VivadoHLSEmitter::emitAffineYield(affine::AffineYieldOp op) {
  if (op->getNumOperands() == 0)
    return;

  emitYieldAssignments(op->getParentOp(), op->getOperands());
}

void VivadoHLSEmitter::emitAffineApply(affine::AffineApplyOp op) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op.getResult());
  os << " = ";
  AffineExprEmitter exprEmitter(state, op.getMapOperands(),
                                op.getAffineMap().getNumDims());
  exprEmitter.emitAffineMap(op.getAffineMap());
  os << ";";
}

void VivadoHLSEmitter::emitBlock(Block &block) {
  for (auto &op : block.getOperations()) {
    dispatch(&op);
  }
}

void VivadoHLSEmitter::emitArraySuffix(ArrayRef<int64_t> shape, Location loc) {
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim)) {
      emitError(loc)
          << "Dynamic shaped types are not supported in Vivado HLS emitter.";
      state.failed = true;
      return;
    }
    state.os << "[" << dim << "]";
  }
}

void VivadoHLSEmitter::emitArraySuffix(ShapedType type, Location loc) {
  if (!type.hasRank()) {
    emitError(loc)
        << "Unranked shaped types are not supported in Vivado HLS emitter.";
    state.failed = true;
    return;
  }
  emitArraySuffix(type.getShape(), loc);
}

void VivadoHLSEmitter::emitValueDecl(Value val, bool isSigned) {
  if (state.hasName(val)) {
    state.os << state.getName(val);
    return;
  }

  if (auto streamType = dyn_cast<StreamType>(val.getType())) {
    state.os << getStreamTypeName(streamType) << " " << state.addName(val);
    emitArraySuffix(streamType.getShape(), val.getLoc());
    return;
  }

  state.os << getPrimitiveTypeName(val.getType(), isSigned) << " "
           << state.addName(val);
  if (auto shaped = dyn_cast<ShapedType>(val.getType()))
    emitArraySuffix(shaped, val.getLoc());
}

void VivadoHLSEmitter::emitValueRef(Value val) {
  if (!state.hasName(val)) {
    emitError(val.getLoc()) << "value used before declaration in Vivado HLS "
                               "emitter.";
    state.failed = true;
    state.os << "/*unknown*/";
    return;
  }
  state.os << state.getName(val);
}

// Emit ``static_cast<T>(value)`` where T is value's own type rendered with the
// requested signedness. MLIR integers are signless and locals default to the
// unsigned C++ type, so sign-sensitive ops route operands through this to read
// them with the signedness their semantics dictate.
void VivadoHLSEmitter::emitSignedOperand(Value value, bool isSigned) {
  state.os << "static_cast<" << getPrimitiveTypeName(value.getType(), isSigned)
           << ">(";
  emitValueRef(value);
  state.os << ")";
}

void VivadoHLSEmitter::emitIndexedValue(Value value, ValueRange indices) {
  emitValueRef(value);
  for (Value index : indices) {
    state.os << "[";
    emitValueRef(index);
    state.os << "]";
  }
}

// Emit nested loops that move one block between the scalar FIFO `stream` (at
// `streamIndices`) and the local array `valueName`, one element per FIFO
// transaction. `isPut` writes the block into the stream; otherwise it reads it.
void VivadoHLSEmitter::emitStreamTransferLoops(bool isPut, Value stream,
                                               ValueRange streamIndices,
                                               ShapedType blockType,
                                               ArrayRef<std::string> indices,
                                               llvm::StringRef valueName) {
  assert(blockType.hasRank() && "stream block payload must be ranked");
  if (indices.size() == static_cast<size_t>(blockType.getRank())) {
    state.os.indent(state.currentIndent);
    if (isPut) {
      emitIndexedValue(stream, streamIndices);
      state.os << ".write(" << valueName;
      for (const auto &index : indices)
        state.os << "[" << index << "]";
      state.os << ");\n";
    } else {
      state.os << valueName;
      for (const auto &index : indices)
        state.os << "[" << index << "]";
      state.os << " = ";
      emitIndexedValue(stream, streamIndices);
      state.os << ".read();\n";
    }
    return;
  }

  int64_t dim = blockType.getDimSize(indices.size());
  assert(!ShapedType::isDynamic(dim) && "stream block payload must be static");
  std::string iv = getTemporaryName("i");
  state.os.indent(state.currentIndent);
  state.os << "for (" << getIntegerTypeName(state.indexWidth, true) << " " << iv
           << " = 0; " << iv << " < " << dim << "; ++" << iv << ") {\n";
  state.addIndent();
  SmallVector<std::string> nestedIndices(indices.begin(), indices.end());
  nestedIndices.push_back(iv);
  emitStreamTransferLoops(isPut, stream, streamIndices, blockType,
                          nestedIndices, valueName);
  state.reduceIndent();
  state.os.indent(state.currentIndent);
  state.os << "}\n";
}

void VivadoHLSEmitter::emitStreamCreate(allo::StreamCreateOp op) {
  llvm::raw_ostream &os = state.os;
  auto streamType = cast<StreamType>(op.getStream().getType());
  emitValueDecl(op.getStream());
  os << ";\n";
  os.indent(state.currentIndent);
  os << "#pragma HLS stream variable=" << state.getName(op.getStream())
     << " depth=" << streamFifoDepth(streamType);
}

void VivadoHLSEmitter::emitStreamGet(allo::StreamGetOp op) {
  llvm::raw_ostream &os = state.os;
  auto streamType = cast<StreamType>(op.getStream().getType());
  if (auto blockType = dyn_cast<ShapedType>(streamType.getBaseType())) {
    emitValueDecl(op.getValue());
    os << ";\n";
    os.indent(state.currentIndent);
    os << "{\n";
    state.addIndent();
    emitStreamTransferLoops(/*isPut=*/false, op.getStream(), op.getIndices(),
                            blockType, {}, state.getName(op.getValue()));
    state.reduceIndent();
    os.indent(state.currentIndent);
    os << "}";
    return;
  }

  emitValueDecl(op.getValue());
  os << " = ";
  emitIndexedValue(op.getStream(), op.getIndices());
  os << ".read();";
}

void VivadoHLSEmitter::emitStreamPut(allo::StreamPutOp op) {
  llvm::raw_ostream &os = state.os;
  auto streamType = cast<StreamType>(op.getStream().getType());
  if (auto blockType = dyn_cast<ShapedType>(streamType.getBaseType())) {
    os << "{\n";
    state.addIndent();
    emitStreamTransferLoops(/*isPut=*/true, op.getStream(), op.getIndices(),
                            blockType, {}, state.getName(op.getValue()));
    state.reduceIndent();
    os.indent(state.currentIndent);
    os << "}";
    return;
  }

  emitIndexedValue(op.getStream(), op.getIndices());
  os << ".write(";
  emitValueRef(op.getValue());
  os << ");";
}

void VivadoHLSEmitter::emitBitGetSlice(allo::BitGetSliceOp op) {
  unsigned width = cast<IntegerType>(op.getResult().getType()).getWidth();
  if (width > 64) {
    op->emitError() << "Bit slices wider than 64 bits are not supported in "
                       "Vivado HLS emitter.";
    state.failed = true;
    return;
  }
  // result = (src >> lo) & mask, where the static width fixes the mask. The
  // offset `lo` may be dynamic, so the shift handles it directly.
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op.getResult());
  os << " = (";
  emitValueRef(op.getSrc());
  os << " >> ";
  emitValueRef(op.getLo());
  os << ") & " << getBitMaskLiteral(width) << ";";
}

void VivadoHLSEmitter::emitBitSetSlice(allo::BitSetSliceOp op) {
  unsigned width = cast<IntegerType>(op.getValue().getType()).getWidth();
  if (width > 64) {
    op->emitError() << "Bit slices wider than 64 bits are not supported in "
                       "Vivado HLS emitter.";
    state.failed = true;
    return;
  }
  // result = (src & ~(mask << lo)) | ((value & mask) << lo): clear the target
  // window in `src`, then splice in the masked value at the (possibly dynamic)
  // offset `lo`.
  std::string mask = getBitMaskLiteral(width);
  std::string srcType = getPrimitiveTypeName(op.getSrc().getType());
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op.getResult());
  os << " = (";
  emitValueRef(op.getSrc());
  os << " & ~(" << mask << " << ";
  emitValueRef(op.getLo());
  os << ")) | ((static_cast<" << srcType << ">(";
  emitValueRef(op.getValue());
  os << ") & " << mask << ") << ";
  emitValueRef(op.getLo());
  os << ");";
}

void VivadoHLSEmitter::emitYieldAssignments(Operation *parent,
                                            OperandRange operands) {
  llvm::raw_ostream &os = state.os;
  unsigned cnt = 0;
  unsigned nResults = parent->getNumResults();
  for (auto [iter, operand] : llvm::zip(parent->getResults(), operands)) {
    emitValueRef(iter);
    os << " = ";
    emitValueRef(operand);
    os << ";";
    if (++cnt != nResults) {
      os << "\n";
      os.indent(state.currentIndent);
    }
  }
}

void VivadoHLSEmitter::emitAffineMapReduction(
    AffineMap map, OperandRange operands, llvm::StringLiteral functionName) {
  assert(map.getNumResults() > 0 && "expected affine map result");
  AffineExprEmitter emitter(state, operands, map.getNumDims());
  if (map.getNumResults() == 1) {
    emitter.visit(map.getResult(0));
    return;
  }
  for (unsigned i = 0; i + 1 < map.getNumResults(); ++i) {
    state.os << functionName << "(";
    emitter.visit(map.getResult(i));
    state.os << ", ";
  }
  emitter.visit(map.getResult(map.getNumResults() - 1));
  for (unsigned i = 0; i + 1 < map.getNumResults(); ++i)
    state.os << ")";
}

void VivadoHLSEmitter::emitMemrefAlloc(memref::AllocOp op) {
  llvm::raw_ostream &os = state.os;
  // A generated temporary may carry an `allo.signed` marker so its element type
  // matches the signedness of the callee it feeds (see
  // materialize-apint-wrapper).
  bool isSigned = false;
  if (auto attr = op->getAttrOfType<StringAttr>("allo.signed"))
    isSigned = attr.getValue() == "s";
  emitValueDecl(op.getResult(), isSigned);
  os << ";";
  // A local on-chip buffer may carry an `allo.part` attribute from a scheduled
  // array_partition (e.g. reuse buffers that must feed a systolic array at full
  // bandwidth). Emit the matching pragma -- the arg path above only covers
  // function arguments.
  if (auto partAttr = op->getAttrOfType<allo::PartitionAttr>("allo.part")) {
    os << "\n";
    emitPartitionAttr(partAttr, op.getResult());
  }
}

void VivadoHLSEmitter::emitMemrefAlloca(memref::AllocaOp op) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op.getResult());
  os << ";";
}

void VivadoHLSEmitter::emitMemrefLoad(memref::LoadOp op) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op.getResult());
  os << " = ";
  emitIndexedValue(op.getMemref(), op.getIndices());
  os << ";";
}

void VivadoHLSEmitter::emitMemrefStore(memref::StoreOp op) {
  llvm::raw_ostream &os = state.os;
  emitIndexedValue(op.getMemref(), op.getIndices());
  os << " = ";
  emitValueRef(op.getValueToStore());
  os << ";";
}

void VivadoHLSEmitter::emitMemrefGlobal(memref::GlobalOp op) {
  llvm::raw_ostream &os = state.os;
  auto type = cast<MemRefType>(op.getType());
  auto initValue = op.getInitialValue();
  auto dense =
      initValue ? dyn_cast<DenseElementsAttr>(*initValue) : DenseElementsAttr();
  // A global with a constant initializer is defined at file scope (internal
  // linkage) so its initial value -- and any state it accumulates across
  // top-function calls -- survives into csim and synthesis. A global without an
  // initializer stays an `extern` declaration defined in another translation
  // unit. `emitModule` emits all globals before any function, so the definition
  // is always in scope at the point of use.
  if (!dense) {
    os << "extern " << getPrimitiveTypeName(type) << " "
       << getSymbolName(op.getSymName());
    emitArraySuffix(type, op.getLoc());
    os << ";";
    return;
  }
  os << "static " << getPrimitiveTypeName(type) << " "
     << getSymbolName(op.getSymName());
  emitArraySuffix(type, op.getLoc());
  os << " = ";
  emitDenseInitializer(dense, type);
  os << ";";
}

void VivadoHLSEmitter::emitDenseInitializer(DenseElementsAttr dense,
                                            MemRefType type) {
  llvm::raw_ostream &os = state.os;
  // Rank-0 memrefs are scalars (`T x = v;`); ranked ones are C arrays whose
  // aggregate initializer is a flat brace list (`T x[..] = {v0, v1, ...};`).
  bool isArray = type.getRank() > 0;
  Type elemType = type.getElementType();
  if (isArray)
    os << "{";
  bool first = true;
  if (isa<IntegerType, IndexType>(elemType)) {
    unsigned width =
        isa<IndexType>(elemType) ? 64 : cast<IntegerType>(elemType).getWidth();
    if (width > 64) {
      emitError(UnknownLoc::get(elemType.getContext()))
          << "global initializer wider than 64 bits is not supported";
      state.failed = true;
      return;
    }
    // Globals print as unsigned C types (getPrimitiveTypeName default), so emit
    // the zero-extended value with a matching unsigned suffix; signedness is
    // recovered by the static_cast the emitter already inserts on each load.
    const char *suffix = width <= 32 ? "u" : "ull";
    for (const APInt &v : dense.getValues<APInt>()) {
      os << (first ? "" : ", ") << v.getZExtValue() << suffix;
      first = false;
    }
  } else if (isa<FloatType>(elemType)) {
    for (const APFloat &v : dense.getValues<APFloat>()) {
      os << (first ? "" : ", ") << v.convertToDouble();
      first = false;
    }
  } else {
    emitError(UnknownLoc::get(elemType.getContext()))
        << "unsupported global initializer element type in Vivado HLS emitter";
    state.failed = true;
  }
  if (isArray)
    os << "}";
}

void VivadoHLSEmitter::emitMemrefGetGlobal(memref::GetGlobalOp op) {
  // we only need to map the result of get_global to the global variable name
  state.nameTable[op.getResult()] = getSymbolName(op.getName());
}

void VivadoHLSEmitter::emitFor(scf::ForOp op) {
  llvm::raw_ostream &os = state.os;
  // declare variables for iter args
  for (auto [result, iter, init] :
       llvm::zip(op.getResults(), op.getRegionIterArgs(), op.getInits())) {
    emitValueDecl(iter);
    os << " = ";
    emitValueRef(init);
    os << ";\n";
    state.nameTable[result] = state.getName(iter);
  }

  if (op.getNumResults())
    os.indent(state.currentIndent);
  os << "for (";
  emitValueDecl(op.getInductionVar());
  os << " = ";
  emitValueRef(op.getLowerBound());
  os << "; " << state.getName(op.getInductionVar()) << " < ";
  emitValueRef(op.getUpperBound());
  os << "; " << state.getName(op.getInductionVar()) << " += ";
  emitValueRef(op.getStep());
  os << ") {\n";
  state.addIndent();
  // emit pragmas
  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  state.reduceIndent();
  os.indent(state.currentIndent);
  os << "}";
}

void VivadoHLSEmitter::emitIf(scf::IfOp op) {
  llvm::raw_ostream &os = state.os;
  // emit results
  for (auto result : op.getResults()) {
    emitValueDecl(result);
    os << ";\n"; // leave it unintialized for now, will be assigned in the
                 // then/else blocks
  }
  if (op.getNumResults())
    os.indent(state.currentIndent);
  os << "if (";
  emitValueRef(op.getCondition());
  os << ") {\n";
  state.addIndent();
  emitBlock(*op.thenBlock());
  state.reduceIndent();
  os.indent(state.currentIndent);
  os << "}";

  if (op.elseBlock() != nullptr) {
    os << " else {\n";
    state.addIndent();
    emitBlock(*op.elseBlock());
    state.reduceIndent();
    os.indent(state.currentIndent);
    os << "}";
  }
}

void VivadoHLSEmitter::emitSCFYield(scf::YieldOp op) {
  if (op->getNumOperands() == 0)
    return;
  emitYieldAssignments(op->getParentOp(), op->getOperands());
}

void VivadoHLSEmitter::emitCastOp(Operation *op) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op->getResult(0));
  os << " = static_cast<" << getPrimitiveTypeName(op->getResult(0).getType())
     << ">(";
  emitValueRef(op->getOperand(0));
  os << ");";
}

// Integer widening. MLIR integers are signless, so route the operand through
// its correctly-signed C++ type to get sign- (extsi) or zero- (extui)
// extension; a plain cast to the default-unsigned result type would always
// zero-extend.
void VivadoHLSEmitter::emitIntExtOp(Operation *op, bool isSigned) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op->getResult(0), isSigned);
  os << " = static_cast<"
     << getPrimitiveTypeName(op->getResult(0).getType(), isSigned) << ">(";
  emitSignedOperand(op->getOperand(0), isSigned);
  os << ");";
}

// arith fp-to-int (fptosi/fptoui): the integer result's signedness is fixed by
// the op. Route the result through that signed/unsigned type so a negative
// float yields the right two's-complement bits rather than an unsigned
// conversion, which is undefined for negatives.
void VivadoHLSEmitter::emitFPToIntOp(Operation *op, bool isSigned) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op->getResult(0), isSigned);
  os << " = static_cast<"
     << getPrimitiveTypeName(op->getResult(0).getType(), isSigned) << ">(";
  emitValueRef(op->getOperand(0));
  os << ");";
}

// arith int-to-fp (sitofp/uitofp): the integer operand's signedness is fixed by
// the op. Read the signless operand through that integer type before
// converting, else a negative value would convert as a large unsigned one.
void VivadoHLSEmitter::emitIntToFPOp(Operation *op, bool isSigned) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op->getResult(0));
  os << " = static_cast<" << getPrimitiveTypeName(op->getResult(0).getType())
     << ">(";
  emitSignedOperand(op->getOperand(0), isSigned);
  os << ");";
}

// Native C++ scalar types have constexpr constructors; ap_int/ap_fixed/half do
// not, so their constants must be `const` rather than `constexpr`.
static bool hasConstexprCtor(Type t) {
  if (auto it = dyn_cast<IntegerType>(t)) {
    unsigned w = it.getWidth();
    return w == 1 || w == 8 || w == 16 || w == 32 || w == 64;
  }
  return isa<IndexType, Float32Type, Float64Type>(t);
}

void VivadoHLSEmitter::emitConstant(arith::ConstantOp op) {
  state.os << (hasConstexprCtor(op.getResult().getType()) ? "constexpr "
                                                          : "const ");
  emitValueDecl(op.getResult());
  state.os << " = ";
  if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
    state.os << intAttr.getInt();
  } else if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
    state.os << floatAttr.getValueAsDouble();
  } else {
    op->emitError() << "unsupported constant attribute in Vivado HLS emitter.";
    state.failed = true;
    state.os << "0";
  }
  state.os << ";";
}

void VivadoHLSEmitter::emitSelect(arith::SelectOp op) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op.getResult());
  os << " = ";
  emitValueRef(op.getCondition());
  os << " ? ";
  emitValueRef(op.getTrueValue());
  os << " : ";
  emitValueRef(op.getFalseValue());
  os << ";";
}

void VivadoHLSEmitter::emitWhile(scf::WhileOp op) {
  llvm::raw_ostream &os = state.os;
  // emit results
  // declare variables for iter args
  bool emittedIterInit = false;
  for (auto [iter, init] : llvm::zip(op.getResults(), op.getInits())) {
    emitValueDecl(iter);
    os << " = ";
    emitValueRef(init);
    os << ";\n";
    emittedIterInit = true;
  }
  if (emittedIterInit)
    os.indent(state.currentIndent);
  os << "while (true) {\n";
  state.addIndent();
  // construct before block
  emitBlock(*op.getBeforeBody());
  // evaluate condition
  os.indent(state.currentIndent);
  os << "if (!(";
  emitValueRef(op.getConditionOp().getCondition());
  os << "))\n";
  os.indent(state.currentIndent + state.indentSize);
  os << "break;\n";
  emitBlock(*op.getAfterBody());
  state.reduceIndent();
  os.indent(state.currentIndent);
  os << "}";
}

void VivadoHLSEmitter::emitReturn(func::ReturnOp op) {
  llvm::raw_ostream &os = state.os;
  if (op.getNumOperands() > 1) {
    op->emitError()
        << "Multiple return operands are not supported in Vivado HLS emitter.";
    state.failed = true;
    return;
  }
  os << "return";
  if (op.getNumOperands() > 0) {
    os << " ";
    emitValueRef(op.getOperand(0));
  }
  os << ";";
}

void VivadoHLSEmitter::dispatch(Operation *op) {
  if ((isa<scf::YieldOp, affine::AffineYieldOp>(op) &&
       op->getNumOperands() == 0) ||
      isa<scf::ConditionOp>(op)) {
    // Skip terminators that do not materialize as standalone statements.
    return;
  }

  state.os.indent(state.currentIndent);

  llvm::TypeSwitch<Operation *, void>(op)
      // binary ops
      .Case<arith::AddIOp>([&](auto op) { emitBinaryOp(op, "+"); })
      .Case<arith::AddFOp>([&](auto op) { emitBinaryOp(op, "+"); })
      .Case<arith::SubIOp>([&](auto op) { emitBinaryOp(op, "-"); })
      .Case<arith::SubFOp>([&](auto op) { emitBinaryOp(op, "-"); })
      .Case<arith::MulIOp>([&](auto op) { emitBinaryOp(op, "*"); })
      .Case<arith::MulFOp>([&](auto op) { emitBinaryOp(op, "*"); })
      .Case<arith::DivFOp>([&](auto op) { emitBinaryOp(op, "/"); })
      .Case<arith::DivUIOp>([&](auto op) { emitBinaryOp(op, "/", false); })
      .Case<arith::DivSIOp>([&](auto op) { emitBinaryOp(op, "/", true); })
      .Case<arith::RemSIOp>([&](auto op) { emitBinaryOp(op, "%", true); })
      .Case<arith::RemUIOp>([&](auto op) { emitBinaryOp(op, "%", false); })
      .Case<arith::RemFOp>([&](auto op) { emitPrefixBinaryOp(op, "fmod"); })
      .Case<arith::AndIOp>([&](auto op) { emitBinaryOp(op, "&"); })
      .Case<arith::OrIOp>([&](auto op) { emitBinaryOp(op, "|"); })
      .Case<arith::XOrIOp>([&](auto op) { emitBinaryOp(op, "^"); })
      .Case<arith::ShLIOp>([&](auto op) { emitBinaryOp(op, "<<"); })
      .Case<arith::ShRUIOp>([&](auto op) { emitBinaryOp(op, ">>", false); })
      .Case<arith::ShRSIOp>([&](auto op) { emitBinaryOp(op, ">>", true); })
      // Vitis has no ceildiv/floordiv

      // max/min ops: signless integer values default to unsigned C++ types, so
      // cast operands to match the op's signedness -- otherwise a signed maxsi
      // on a negative value would compare as unsigned.
      .Case<arith::MaxSIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "std::max", true); })
      .Case<arith::MinSIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "std::min", true); })
      .Case<arith::MaxUIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "std::max", false); })
      .Case<arith::MinUIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "std::min", false); })
      .Case<arith::MaximumFOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "hls::fmax"); })
      .Case<arith::MinimumFOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "hls::fmin"); })
      .Case<arith::MaxNumFOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "hls::fmax"); })
      .Case<arith::MinNumFOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "hls::fmin"); })

      // unary ops
      .Case<arith::NegFOp>([&](auto op) { emitUnaryOp(op, "-"); })
      .Case<math::AbsIOp>([&](auto op) { emitUnaryOp(op, "hls::abs"); })
      .Case<math::AbsFOp>([&](auto op) { emitUnaryOp(op, "hls::fabs"); })
      .Case<math::ExpOp>([&](auto op) { emitUnaryOp(op, "hls::exp"); })
      .Case<math::Exp2Op>([&](auto op) { emitUnaryOp(op, "hls::exp2"); })
      .Case<math::LogOp>([&](auto op) { emitUnaryOp(op, "hls::log"); })
      .Case<math::Log2Op>([&](auto op) { emitUnaryOp(op, "hls::log2"); })
      .Case<math::Log10Op>([&](auto op) { emitUnaryOp(op, "hls::log10"); })
      .Case<math::SqrtOp>([&](auto op) { emitUnaryOp(op, "hls::sqrt"); })
      .Case<math::RsqrtOp>([&](auto op) { emitUnaryOp(op, "hls::rsqrt"); })
      .Case<math::SinOp>([&](auto op) { emitUnaryOp(op, "hls::sin"); })
      .Case<math::CosOp>([&](auto op) { emitUnaryOp(op, "hls::cos"); })
      .Case<math::TanOp>([&](auto op) { emitUnaryOp(op, "hls::tan"); })
      .Case<math::SinhOp>([&](auto op) { emitUnaryOp(op, "hls::sinh"); })
      .Case<math::CoshOp>([&](auto op) { emitUnaryOp(op, "hls::cosh"); })
      .Case<math::TanhOp>([&](auto op) { emitUnaryOp(op, "hls::tanh"); })
      .Case<math::PowFOp>([&](auto op) { emitPrefixBinaryOp(op, "hls::powf"); })
      .Case<math::IPowIOp>([&](auto op) { emitPrefixBinaryOp(op, "hls::pow"); })
      .Case<math::FPowIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "hls::pown"); })
      .Case<math::FmaOp>([&](auto op) { emitPrefixBinaryOp(op, "hls::fma"); })
      .Case<math::FloorOp>([&](auto op) { emitUnaryOp(op, "hls::floor"); })
      .Case<math::CeilOp>([&](auto op) { emitUnaryOp(op, "hls::ceil"); })
      .Case<math::TruncOp>([&](auto op) { emitUnaryOp(op, "hls::trunc"); })
      .Case<math::RoundOp>([&](auto op) { emitUnaryOp(op, "hls::round"); })

      // cast ops
      .Case<arith::ExtSIOp>(
          [&](auto op) { emitIntExtOp(op, /*isSigned=*/true); })
      .Case<arith::ExtUIOp>(
          [&](auto op) { emitIntExtOp(op, /*isSigned=*/false); })
      // index_cast is the signed integer-resize variant, same shape as extsi.
      .Case<arith::IndexCastOp>(
          [&](auto op) { emitIntExtOp(op, /*isSigned=*/true); })
      .Case<arith::FPToSIOp>(
          [&](auto op) { emitFPToIntOp(op, /*isSigned=*/true); })
      .Case<arith::FPToUIOp>(
          [&](auto op) { emitFPToIntOp(op, /*isSigned=*/false); })
      .Case<arith::SIToFPOp>(
          [&](auto op) { emitIntToFPOp(op, /*isSigned=*/true); })
      .Case<arith::UIToFPOp>(
          [&](auto op) { emitIntToFPOp(op, /*isSigned=*/false); })
      // sign-agnostic: float resize and integer truncation keep low bits.
      .Case<arith::ExtFOp, arith::TruncIOp, arith::TruncFOp>(
          [&](auto op) { emitCastOp(op); })

      // special ops
      .Case<affine::AffineForOp>([&](auto op) { emitAffineFor(op); })
      .Case<affine::AffineLoadOp>([&](auto op) { emitAffineLoad(op); })
      .Case<affine::AffineStoreOp>([&](auto op) { emitAffineStore(op); })
      .Case<affine::AffineYieldOp>([&](auto op) { emitAffineYield(op); })
      .Case<affine::AffineIfOp>([&](auto op) { emitAffineIf(op); })
      .Case<affine::AffineApplyOp>([&](auto op) { emitAffineApply(op); })

      .Case<func::FuncOp>([&](auto op) { emitFunction(op); })
      .Case<func::CallOp>([&](auto op) { emitCall(op); })
      .Case<func::ReturnOp>([&](auto op) { emitReturn(op); })

      .Case<memref::AllocOp>([&](auto op) { emitMemrefAlloc(op); })
      .Case<memref::AllocaOp>([&](auto op) { emitMemrefAlloca(op); })
      // Local arrays free with their scope in C++; dealloc emits nothing.
      .Case<memref::DeallocOp>([&](auto) {})
      .Case<memref::LoadOp>([&](auto op) { emitMemrefLoad(op); })
      .Case<memref::StoreOp>([&](auto op) { emitMemrefStore(op); })
      .Case<memref::GlobalOp>([&](auto op) { emitMemrefGlobal(op); })
      .Case<memref::GetGlobalOp>([&](auto op) { emitMemrefGetGlobal(op); })

      .Case<allo::StreamCreateOp>([&](auto op) { emitStreamCreate(op); })
      .Case<allo::StreamGetOp>([&](auto op) { emitStreamGet(op); })
      .Case<allo::StreamPutOp>([&](auto op) { emitStreamPut(op); })
      .Case<allo::BitGetSliceOp>([&](auto op) { emitBitGetSlice(op); })
      .Case<allo::BitSetSliceOp>([&](auto op) { emitBitSetSlice(op); })

      .Case<arith::ConstantOp>([&](auto op) { emitConstant(op); })
      .Case<arith::SelectOp>([&](auto op) { emitSelect(op); })
      .Case<arith::CmpIOp>([&](auto op) { emitCmpI(op); })
      .Case<arith::CmpFOp>([&](auto op) { emitCmpF(op); })

      .Case<scf::ForOp>([&](auto op) { emitFor(op); })
      .Case<scf::IfOp>([&](auto op) { emitIf(op); })
      .Case<scf::YieldOp>([&](auto op) { emitSCFYield(op); })
      .Case<scf::WhileOp>([&](auto op) { emitWhile(op); })

      .Default([&](auto op) {
        op->emitError() << "operation not supported in Vivado HLS emitter: "
                        << op->getName();
        state.failed = true;
      });

  emitTrailingLocation(op);
}

void VivadoHLSEmitter::emitBinaryOp(Operation *op,
                                    llvm::StringLiteral keyword) {
  llvm::raw_ostream &os = state.os;
  Value result = op->getResult(0);
  emitValueDecl(result);
  os << " = ";
  emitValueRef(op->getOperand(0));
  os << " " << keyword << " ";
  emitValueRef(op->getOperand(1));
  os << ";";
}

void VivadoHLSEmitter::emitBinaryOp(Operation *op, llvm::StringLiteral keyword,
                                    bool isSigned) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op->getResult(0));
  os << " = ";
  emitSignedOperand(op->getOperand(0), isSigned);
  os << " " << keyword << " ";
  emitSignedOperand(op->getOperand(1), isSigned);
  os << ";";
}

void VivadoHLSEmitter::emitUnaryOp(Operation *op, llvm::StringLiteral keyword) {
  llvm::raw_ostream &os = state.os;
  Value result = op->getResult(0);
  emitValueDecl(result);
  os << " = " << keyword << "(";
  emitValueRef(op->getOperand(0));
  os << ");";
}

void VivadoHLSEmitter::emitPrefixBinaryOp(Operation *op,
                                          llvm::StringLiteral keyword) {
  llvm::raw_ostream &os = state.os;
  Value result = op->getResult(0);
  emitValueDecl(result);
  os << " = " << keyword << "(";
  emitValueRef(op->getOperand(0));
  os << ", ";
  emitValueRef(op->getOperand(1));
  os << ");";
}

void VivadoHLSEmitter::emitPrefixBinaryOp(Operation *op,
                                          llvm::StringLiteral keyword,
                                          bool isSigned) {
  llvm::raw_ostream &os = state.os;
  emitValueDecl(op->getResult(0));
  os << " = " << keyword << "(";
  emitSignedOperand(op->getOperand(0), isSigned);
  os << ", ";
  emitSignedOperand(op->getOperand(1), isSigned);
  os << ");";
}

// Only the operator; the signed/unsigned distinction is applied by casting the
// operands in emitCmpI, so slt and ult share "<", etc.
static std::string getCmpIPredString(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return "==";
  case arith::CmpIPredicate::ne:
    return "!=";
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return "<";
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return "<=";
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return ">";
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return ">=";
  }
  llvm_unreachable("unsupported integer comparison predicate");
}

// Ordered signed predicates, whose operands must be read as signed. eq/ne are
// sign-agnostic and unsigned predicates need an unsigned read.
static bool isSignedCmpIPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::sge:
    return true;
  default:
    return false;
  }
}

static std::string getCmpFPredString(arith::CmpFPredicate pred) {
  switch (pred) {
  case arith::CmpFPredicate::OEQ:
    return "==";
  case arith::CmpFPredicate::OGT:
    return ">";
  case arith::CmpFPredicate::OGE:
    return ">=";
  case arith::CmpFPredicate::OLT:
    return "<";
  case arith::CmpFPredicate::OLE:
    return "<=";
  case arith::CmpFPredicate::ONE:
    return "!=";
  case arith::CmpFPredicate::UEQ:
    return "==";
  case arith::CmpFPredicate::UGT:
    return ">";
  case arith::CmpFPredicate::UGE:
    return ">=";
  case arith::CmpFPredicate::ULT:
    return "<";
  case arith::CmpFPredicate::ULE:
    return "<=";
  case arith::CmpFPredicate::UNE:
    return "!=";
  default:
    llvm_unreachable("unsupported floating-point comparison predicate");
  }
}

void VivadoHLSEmitter::emitCmpI(arith::CmpIOp op) {
  llvm::raw_ostream &os = state.os;
  arith::CmpIPredicate pred = op.getPredicate();
  emitValueDecl(op.getResult());
  os << " = ";
  // eq/ne give the same result for either signedness; ordered comparisons do
  // not, so cast the signless operands to the predicate's signedness -- without
  // it a signed slt on a negative (default-unsigned) value compares as
  // unsigned.
  if (pred == arith::CmpIPredicate::eq || pred == arith::CmpIPredicate::ne) {
    emitValueRef(op.getLhs());
    os << " " << getCmpIPredString(pred) << " ";
    emitValueRef(op.getRhs());
  } else {
    bool isSigned = isSignedCmpIPredicate(pred);
    emitSignedOperand(op.getLhs(), isSigned);
    os << " " << getCmpIPredString(pred) << " ";
    emitSignedOperand(op.getRhs(), isSigned);
  }
  os << ";";
}

void VivadoHLSEmitter::emitCmpF(arith::CmpFOp op) {
  llvm::raw_ostream &os = state.os;
  Value result = op.getResult();
  emitValueDecl(result);
  os << " = ";
  emitValueRef(op.getLhs());
  os << " " << getCmpFPredString(op.getPredicate()) << " ";
  emitValueRef(op.getRhs());
  os << ";";
}

constexpr llvm::StringLiteral deviceHeader = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

void VivadoHLSEmitter::emitModule(ModuleOp mod) {
  // TODO: add host-side codegen
  llvm::raw_ostream &os = state.os;

  os << deviceHeader << "\n";
  // Step 1: emit top-level declarations other than functions.
  for (Operation &op : mod.getBody()->without_terminator()) {
    if (isa<func::FuncOp>(&op))
      continue;
    dispatch(&op);
  }

  // Step 2: generate all function declarations
  for (auto func : mod.getOps<func::FuncOp>()) {
    emitFunctionSignature(func);
    os << ";";
    emitTrailingLocation(func);
    os << "\n";
  }

  // Step 3: emit function definitions
  for (auto func : mod.getOps<func::FuncOp>()) {
    emitFunction(func);
    os << "\n";
  }
}

static llvm::cl::opt<unsigned>
    indexWidth("index-width",
               llvm::cl::desc("Bit width to use for index types (default: 32)"),
               llvm::cl::init(32));

static llvm::cl::opt<unsigned>
    indent("indent",
           llvm::cl::desc("Indent width for code generation (default: 2)"),
           llvm::cl::init(2));

static llvm::cl::opt<bool>
    withLocation("with-location",
                 llvm::cl::desc("Include location info as comments in the "
                                "generated code"),
                 llvm::cl::init(false));

static LogicalResult emitVivadoHLS(ModuleOp mod, llvm::raw_ostream &os) {
  return emitVivadoHLS(mod, os, indexWidth, indent, withLocation);
}

LogicalResult allo::emitVivadoHLS(ModuleOp mod, llvm::raw_ostream &os,
                                  unsigned indexWidth, unsigned indentSize,
                                  bool withLocation) {
  VivadoHLSEmitter emitter(os);
  emitter.state.indexWidth = indexWidth;
  emitter.state.indentSize = indentSize;
  emitter.state.withLocation = withLocation;
  emitter.emitModule(mod);
  return failure(emitter.state.failed);
}

void allo::registerVivadoHLSTranslation() {
  static TranslateFromMLIRRegistration reg(
      "emit-vitis-hls", "Translate MLIR to C++ code for Vivado HLS",
      ::emitVivadoHLS, [&](DialectRegistry &registry) {
        registry
            .insert<affine::AffineDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    func::FuncDialect, allo::AlloDialect>();
      });
}
