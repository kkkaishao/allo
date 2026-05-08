/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"
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

bool VivadoHLSEmitter::hasUnsupportedType(Type type) {
  if (isa<StreamType>(type))
    return true;
  if (auto shaped = dyn_cast<ShapedType>(type))
    return hasUnsupportedType(shaped.getElementType());
  return false;
}

LogicalResult VivadoHLSEmitter::validateModule(ModuleOp mod) {
  WalkResult result = mod->walk([&](Operation *op) -> WalkResult {
    if (isa<allo::StreamCreateOp, allo::StreamGetOp, allo::StreamPutOp>(op)) {
      op->emitError()
          << "Stream operations are not supported in Vivado HLS emitter.";
      state.failed = true;
      return WalkResult::interrupt();
    }

    if (auto func = dyn_cast<func::FuncOp>(op)) {
      for (Type type : func.getArgumentTypes()) {
        if (hasUnsupportedType(type)) {
          func->emitError()
              << "StreamType is not supported in Vivado HLS emitter.";
          state.failed = true;
          return WalkResult::interrupt();
        }
      }
      for (Type type : func.getResultTypes()) {
        if (hasUnsupportedType(type)) {
          func->emitError()
              << "StreamType is not supported in Vivado HLS emitter.";
          state.failed = true;
          return WalkResult::interrupt();
        }
      }
    }

    for (Type type : op->getResultTypes()) {
      if (hasUnsupportedType(type)) {
        op->emitError() << "StreamType is not supported in Vivado HLS emitter.";
        state.failed = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
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

  llvm_unreachable("unsupported types");
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

  emitFunctionReturnType(func);
  state.os << " " << getSymbolName(func.getSymName()) << "(";
  emitFunctionArguments(func);
  state.os << ") {\n";
  state.addIndent();
  // emit function-level directives
  emitFunctionDirectives(func);
  emitBlock(func.getBlocks().front());
  state.reduceIndent();
  state.os << "}";
}

void VivadoHLSEmitter::emitFunctionReturnType(func::FuncOp func) {
  unsigned nResults = func.getNumResults();
  if (nResults == 0)
    state.os << "void";
  else if (nResults == 1)
    state.os << getPrimitiveTypeName(func.getResultTypes().front());
  else {
    func->emitError()
        << "Multiple return values are not supported in Vivado HLS emitter.";
    state.failed = true;
  }
}

void VivadoHLSEmitter::emitFunctionArguments(func::FuncOp func) {
  for (auto arg : func.getArguments()) {
    if (arg != func.getArguments().front())
      state.os << ", ";
    state.os << getPrimitiveTypeName(arg.getType()) << " "
             << state.getOrAddName(arg);
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

void VivadoHLSEmitter::emitArraySuffix(ShapedType type, Location loc) {
  if (!type.hasRank()) {
    emitError(loc)
        << "Unranked shaped types are not supported in Vivado HLS emitter.";
    state.failed = true;
    return;
  }
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      emitError(loc)
          << "Dynamic shaped types are not supported in Vivado HLS emitter.";
      state.failed = true;
      return;
    }
    state.os << "[" << dim << "]";
  }
}

void VivadoHLSEmitter::emitValueDecl(Value val) {
  if (state.hasName(val)) {
    state.os << state.getName(val);
    return;
  }

  state.os << getPrimitiveTypeName(val.getType()) << " " << state.addName(val);
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

void VivadoHLSEmitter::emitIndexedValue(Value value, ValueRange indices) {
  emitValueRef(value);
  for (Value index : indices) {
    state.os << "[";
    emitValueRef(index);
    state.os << "]";
  }
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
  emitValueDecl(op.getResult());
  os << ";";
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
  // it has a symbol name, we can use it directly
  os << "extern ";
  auto type = cast<MemRefType>(op.getType());
  os << getPrimitiveTypeName(type);
  os << " " << getSymbolName(op.getSymName());
  emitArraySuffix(type, op.getLoc());
  os << ";";
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

void VivadoHLSEmitter::emitConstant(arith::ConstantOp op) {
  state.os << "constexpr ";
  emitValueDecl(op.getResult());
  state.os << " = ";
  if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
    state.os << intAttr.getInt();
  } else if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
    state.os << floatAttr.getValueAsDouble();
  } else {
    llvm_unreachable("unsupported constant attribute");
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

void VivadoHLSEmitter::emitLinalgFill(linalg::FillOp op) {
  llvm::raw_ostream &os = state.os;

  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) {
    op->emitError()
        << "Only single-input single-output linalg.fill is supported in "
           "Vivado HLS emitter.";
    state.failed = true;
    return;
  }

  Value fillValue = op.getInputs().front();
  Value output = op.getOutputs().front();
  auto type = dyn_cast<MemRefType>(output.getType());
  if (!type || !type.hasStaticShape()) {
    op->emitError()
        << "Only static ranked memref linalg.fill outputs are supported in "
           "Vivado HLS emitter.";
    state.failed = true;
    return;
  }

  if (type.getRank() == 0) {
    emitValueRef(output);
    os << " = ";
    emitValueRef(fillValue);
    os << ";";
    return;
  }

  std::string indexType = getIntegerTypeName(state.indexWidth, true);
  SmallVector<std::string, 4> loopVars;
  loopVars.reserve(type.getRank());
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    std::string iv = "i" + std::to_string(dim);
    loopVars.push_back(iv);
    os << "for (" << indexType << " " << iv << " = 0; " << iv << " < "
       << type.getDimSize(dim) << "; ++" << iv << ") {\n";
    state.addIndent();
    os.indent(state.currentIndent);
  }

  emitValueRef(output);
  for (const std::string &iv : loopVars)
    os << "[" << iv << "]";
  os << " = ";
  emitValueRef(fillValue);
  os << ";";

  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    state.reduceIndent();
    os << "\n";
    os.indent(state.currentIndent);
    os << "}";
  }
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

      // max/min ops
      .Case<arith::MaxSIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "std::max"); })
      .Case<arith::MinSIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "std::min"); })
      .Case<arith::MaxUIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "std::max"); })
      .Case<arith::MinUIOp>(
          [&](auto op) { emitPrefixBinaryOp(op, "std::min"); })
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
      .Case<math::AbsIOp>([&](auto op) { emitUnaryOp(op, "hls::abs"); })
      .Case<math::AbsFOp>([&](auto op) { emitUnaryOp(op, "hls::fabs"); })
      .Case<math::FloorOp>([&](auto op) { emitUnaryOp(op, "hls::floor"); })
      .Case<math::CeilOp>([&](auto op) { emitUnaryOp(op, "hls::ceil"); })
      .Case<math::TruncOp>([&](auto op) { emitUnaryOp(op, "hls::trunc"); })
      .Case<math::RoundOp>([&](auto op) { emitUnaryOp(op, "hls::round"); })

      // cast ops
      .Case<arith::IndexCastOp, arith::FPToSIOp, arith::FPToUIOp,
            arith::SIToFPOp, arith::UIToFPOp, arith::ExtFOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::TruncIOp, arith::TruncFOp>(
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
      .Case<memref::LoadOp>([&](auto op) { emitMemrefLoad(op); })
      .Case<memref::StoreOp>([&](auto op) { emitMemrefStore(op); })
      .Case<memref::GlobalOp>([&](auto op) { emitMemrefGlobal(op); })
      .Case<memref::GetGlobalOp>([&](auto op) { emitMemrefGetGlobal(op); })

      .Case<arith::ConstantOp>([&](auto op) { emitConstant(op); })
      .Case<arith::SelectOp>([&](auto op) { emitSelect(op); })
      .Case<arith::CmpIOp>([&](auto op) { emitCmpI(op); })
      .Case<arith::CmpFOp>([&](auto op) { emitCmpF(op); })

      .Case<scf::ForOp>([&](auto op) { emitFor(op); })
      .Case<scf::IfOp>([&](auto op) { emitIf(op); })
      .Case<scf::YieldOp>([&](auto op) { emitSCFYield(op); })
      .Case<scf::WhileOp>([&](auto op) { emitWhile(op); })

      .Case<linalg::FillOp>([&](auto op) { emitLinalgFill(op); })

      .Default([&](auto op) {
        op->emitError() << "operation not supported in Vivado HLS emitter: "
                        << op->getName();
        state.failed = true;
      });

  // generate location info
  if (!state.withLocation) {
    state.os << "\n";
    return;
  }
  if (auto loc = dyn_cast<FileLineColLoc>(op->getLoc())) {
    state.os << "\t// " << loc.getFilename() << ":" << loc.getLine() << ":"
             << loc.getColumn() << "\n";
  } else {
    // ensure a new line
    state.os << "\n";
  }
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
  Value result = op->getResult(0);
  emitValueDecl(result);
  os << " = ";
  os << "static_cast<"
     << getPrimitiveTypeName(op->getOperand(0).getType(), isSigned) << ">(";
  emitValueRef(op->getOperand(0));
  os << ") " << keyword << " ";
  os << "static_cast<"
     << getPrimitiveTypeName(op->getOperand(1).getType(), isSigned) << ">(";
  emitValueRef(op->getOperand(1));
  os << ");";
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
  Value result = op->getResult(0);
  emitValueDecl(result);
  os << " = " << keyword << "(";
  os << "static_cast<"
     << getPrimitiveTypeName(op->getOperand(0).getType(), isSigned) << ">(";
  emitValueRef(op->getOperand(0));
  os << "), ";
  os << "static_cast<"
     << getPrimitiveTypeName(op->getOperand(1).getType(), isSigned) << ">(";
  emitValueRef(op->getOperand(1));
  os << "));";
}

static std::string getCmpIPredString(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return "==";
  case arith::CmpIPredicate::ne:
    return "!=";
  case arith::CmpIPredicate::slt:
    return "<";
  case arith::CmpIPredicate::sgt:
    return ">";
  case arith::CmpIPredicate::sle:
    return "<=";
  case arith::CmpIPredicate::sge:
    return ">=";
  case arith::CmpIPredicate::ult:
    return "<";
  case arith::CmpIPredicate::ugt:
    return ">";
  case arith::CmpIPredicate::ule:
    return "<=";
  case arith::CmpIPredicate::uge:
    return ">=";
  default:
    llvm_unreachable("unsupported integer comparison predicate");
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
  Value result = op.getResult();
  emitValueDecl(result);
  os << " = ";
  emitValueRef(op.getLhs());
  os << " " << getCmpIPredString(op.getPredicate()) << " ";
  emitValueRef(op.getRhs());
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
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

void VivadoHLSEmitter::emitModule(ModuleOp mod) {
  // TODO: add host-side codegen
  llvm::raw_ostream &os = state.os;
  if (failed(validateModule(mod)))
    return;

  os << deviceHeader << "\n";
  // Step 1: emit global variables
  mod->walk([&](memref::GlobalOp op) { dispatch(op); });

  // Step 2: generate all function declarations
  for (auto func : mod.getOps<func::FuncOp>()) {
    emitFunctionReturnType(func);
    os << " " << getSymbolName(func.getSymName()) << "(";
    emitFunctionArguments(func);
    os << ");";
    if (state.withLocation) {
      if (auto loc = dyn_cast<FileLineColLoc>(func.getLoc())) {
        state.os << "\t// " << loc.getFilename() << ":" << loc.getLine() << ":"
                 << loc.getColumn() << "\n\n";
      }
    } else
      os << "\n\n";
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
        registry.insert<affine::AffineDialect, arith::ArithDialect,
                        linalg::LinalgDialect, math::MathDialect,
                        memref::MemRefDialect, scf::SCFDialect,
                        func::FuncDialect, allo::AlloDialect>();
      });
}
