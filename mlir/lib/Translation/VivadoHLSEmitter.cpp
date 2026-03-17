/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "allo/Translation/VivadoHLSEmitter.h"

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
    return "ap_" + prefix + "int" + std::to_string(width) + "_t";
  }
}

std::string VivadoHLSEmitter::getPrimitiveTypeName(Type type) {
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
    bool isSigned = intType.isSigned();
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

  if (auto streamType = dyn_cast<StreamType>(type)) {
    // Check if the base type is a shaped type (tensor/array) - stream of blocks
    if (auto shapedType =
            llvm::dyn_cast<ShapedType>(streamType.getBaseType())) {
      // Stream of blocks using hls::vector: Stream[elementType[dims...], depth]
      // Flatten all dimensions into a single vector size
      int64_t vectorSize = std::accumulate(shapedType.getShape().begin(),
                                           shapedType.getShape().end(), 1LL,
                                           std::multiplies<int64_t>());
      std::string elementTypeName =
          getPrimitiveTypeName(shapedType.getElementType());
      return "hls::stream<hls::vector<" + elementTypeName + ", " +
             std::to_string(vectorSize) + ">>";
    }
    return "hls::stream<" + getPrimitiveTypeName(streamType.getBaseType()) +
           ">";
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
  state.os << " " << func.getSymName() << "(";
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
    if (auto shaped = dyn_cast<ShapedType>(arg.getType())) {
      for (auto dim : shaped.getShape()) {
        state.os << "[" << dim << "]";
      }
    }
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
  if (op->getNumResults() == 1) {
    emitValue(op.getResult(0));
    os << " = ";
  }
  os << op.getCallee() << "(";
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (i > 0)
      os << ", ";
    emitValue(op.getOperand(i));
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
    emitValue(iter);
    os << " = ";
    emitValue(init);
    os << ";\n";
    state.nameTable[result] = state.getName(iter);
  }

  if (op.getNumResults())
    os.indent(state.currentIndent);
  os << "for (";
  emitValue(op.getInductionVar());
  os << " = ";
  std::string ivName = state.getName(op.getInductionVar());
  AffineExprEmitter lbEmitter(state, op.getLowerBoundOperands());
  AffineMap lbMap = op.getLowerBoundMap();
  // if lb num results > 1, affine.for will take the max of all results as the
  // lower bound
  if (lbMap.getNumResults() > 1) {
    os << "max(";
    lbEmitter.emitAffineMap(lbMap);
    os << ")";
  } else {
    lbEmitter.emitAffineMap(lbMap);
  }
  // if ub num results > 1, affine.for will take the min of all results as the
  // upper bound
  os << "; " << ivName << " < ";
  AffineExprEmitter ubEmitter(state, op.getUpperBoundOperands());
  AffineMap ubMap = op.getUpperBoundMap();
  if (ubMap.getNumResults() > 1) {
    os << "min(";
    ubEmitter.emitAffineMap(ubMap);
    os << ")";
  } else {
    ubEmitter.emitAffineMap(ubMap);
  }
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
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getMemref());
  AffineExprEmitter indexEmitter(state, op.getMapOperands());
  AffineMap indexMap = op.getAffineMap();
  for (unsigned i = 0; i < indexMap.getNumResults(); ++i) {
    os << "[";
    indexEmitter.visit(indexMap.getResult(i));
    os << "]";
  }
  os << ";";
}

void VivadoHLSEmitter::emitAffineStore(affine::AffineStoreOp op) {
  llvm::raw_ostream &os = state.os;
  emitValue(op.getMemref());
  AffineExprEmitter indexEmitter(state, op.getMapOperands());
  AffineMap indexMap = op.getAffineMap();
  for (unsigned i = 0; i < indexMap.getNumResults(); ++i) {
    os << "[";
    indexEmitter.visit(indexMap.getResult(i));
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
}

void VivadoHLSEmitter::emitAffineIf(affine::AffineIfOp op) {
  llvm::raw_ostream &os = state.os;
  // emit results
  for (auto result : op.getResults()) {
    emitValue(result);
    os << ";\n"; // leave it uninitialized for now, will be assigned in the
                 // then/else blocks
  }
  if (op.getNumResults())
    os.indent(state.currentIndent);
  os << "if (";
  AffineExprEmitter condEmitter(state, op->getOperands());

  IntegerSet conds = op.getCondition();
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

  llvm::raw_ostream &os = state.os;
  auto parent = op->getParentOp();
  unsigned cnt = 0;
  unsigned nResults = parent->getNumResults();
  for (auto [iter, operand] :
       llvm::zip(parent->getResults(), op->getOperands())) {
    emitValue(iter);
    os << " = ";
    emitValue(operand);
    os << ";";
    if (++cnt != nResults) {
      os << "\n";
      os.indent(state.currentIndent);
    }
  }
}

void VivadoHLSEmitter::emitAffineApply(affine::AffineApplyOp op) {
  llvm::raw_ostream &os = state.os;
  emitValue(op.getResult());
  os << " = ";
  AffineExprEmitter exprEmitter(state, op.getMapOperands());
  exprEmitter.emitAffineMap(op.getAffineMap());
  os << ";";
}

void VivadoHLSEmitter::emitBlock(Block &block) {
  for (auto &op : block.getOperations()) {
    dispatch(&op);
  }
}

void VivadoHLSEmitter::emitValue(Value val) {
  // generate type declaration if not declared yet
  if (state.hasName(val)) {
    state.os << state.getName(val);
  } else {
    state.os << getPrimitiveTypeName(val.getType()) << " "
             << state.addName(val);
    if (auto shaped = dyn_cast<ShapedType>(val.getType())) {
      if (!shaped.hasRank()) {
        emitError(val.getLoc())
            << "Unranked shaped types are not supported in Vivado HLS emitter.";
        state.failed = true;
      }
      // if it's a shaped type, we need to declare the shape as well
      for (auto dim : shaped.getShape()) {
        state.os << "[" << dim << "]";
      }
    }
  }
}

void VivadoHLSEmitter::emitMemrefAlloc(memref::AllocOp op) {
  llvm::raw_ostream &os = state.os;
  emitValue(op.getResult());
  os << ";";
}

void VivadoHLSEmitter::emitMemrefAlloca(memref::AllocaOp op) {
  llvm::raw_ostream &os = state.os;
  emitValue(op.getResult());
  os << ";";
}

void VivadoHLSEmitter::emitMemrefLoad(memref::LoadOp op) {
  llvm::raw_ostream &os = state.os;
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getMemref());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
}

void VivadoHLSEmitter::emitMemrefStore(memref::StoreOp op) {
  llvm::raw_ostream &os = state.os;
  emitValue(op.getMemref());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
}

void VivadoHLSEmitter::emitMemrefGlobal(memref::GlobalOp op) {
  llvm::raw_ostream &os = state.os;
  // it has a symbol name, we can use it directly
  os << "extern ";
  auto type = cast<MemRefType>(op.getType());
  os << getPrimitiveTypeName(type);
  os << " " << op.getSymName();
  for (auto dim : type.getShape()) {
    os << "[" << dim << "]";
  }
  os << ";";
}

void VivadoHLSEmitter::emitMemrefGetGlobal(memref::GetGlobalOp op) {
  // we only need to map the result of get_global to the global variable name
  state.nameTable[op.getResult()] = op.getName();
}

void VivadoHLSEmitter::emitFor(scf::ForOp op) {
  llvm::raw_ostream &os = state.os;
  // declare variables for iter args
  for (auto [result, iter, init] :
       llvm::zip(op.getResults(), op.getRegionIterArgs(), op.getInits())) {
    emitValue(iter);
    os << " = ";
    emitValue(init);
    os << ";\n";
    state.nameTable[result] = state.getName(iter);
  }

  if (op.getNumResults())
    os.indent(state.currentIndent);
  os << "for (";
  emitValue(op.getInductionVar());
  os << " = ";
  emitValue(op.getLowerBound());
  os << "; " << state.getName(op.getInductionVar()) << " < ";
  emitValue(op.getUpperBound());
  os << "; " << state.getName(op.getInductionVar()) << " += ";
  emitValue(op.getStep());
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
    emitValue(result);
    os << ";\n"; // leave it unintialized for now, will be assigned in the
                 // then/else blocks
  }
  if (op.getNumResults())
    os.indent(state.currentIndent);
  os << "if (";
  emitValue(op.getCondition());
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
  llvm::raw_ostream &os = state.os;
  Operation *parent = op->getParentOp();
  unsigned cnt = 0;
  unsigned nIterArgs = parent->getNumResults();
  for (auto [iter, operand] :
       llvm::zip(parent->getResults(), op->getOperands())) {
    emitValue(iter);
    os << " = ";
    emitValue(operand);
    os << ";";
    if (++cnt != nIterArgs) {
      os << "\n";
      os.indent(state.currentIndent);
    }
  }
}

void VivadoHLSEmitter::emitCastOp(Operation *op) {
  llvm::raw_ostream &os = state.os;
  emitValue(op->getResult(0));
  os << " = (" << getPrimitiveTypeName(op->getResult(0).getType()) << ")";
  emitValue(op->getOperand(0));
  os << ";";
}

void VivadoHLSEmitter::emitConstant(arith::ConstantOp op) {
  state.os << "constexpr ";
  emitValue(op.getResult());
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
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getCondition());
  os << " ? ";
  emitValue(op.getTrueValue());
  os << " : ";
  emitValue(op.getFalseValue());
  os << ";";
}

void VivadoHLSEmitter::emitWhile(scf::WhileOp op) {
  llvm::raw_ostream &os = state.os;
  // emit results
  // declare variables for iter args
  bool emittedIterInit = false;
  for (auto [iter, init] : llvm::zip(op.getResults(), op.getInits())) {
    emitValue(iter);
    os << " = ";
    emitValue(init);
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
  os << "if (!(";
  emitValue(op.getConditionOp().getCondition());
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
  os << "return";
  if (op.getNumOperands() > 0) {
    os << " ";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (i > 0)
        os << ", ";
      emitValue(op.getOperand(i));
    }
  }
  os << ";";
}

void VivadoHLSEmitter::emitCondition(scf::ConditionOp op) {
  // llvm::raw_ostream &os = state.os;
  // emitValue(op.getCondition());
  // os << " = ";
  // emitValue(op.getOperand());
  // os << ";";
}

void VivadoHLSEmitter::dispatch(Operation *op) {
  if (isa<scf::YieldOp, affine::AffineYieldOp>(op) &&
      op->getNumOperands() == 0) {
    // Skip empty yields to avoid generating blank lines.
    return;
  }

  state.os.indent(state.currentIndent);

  llvm::TypeSwitch<Operation *, void>(op)
      // binary ops
      .Case<arith::AddIOp>([&](auto op) { emitBinaryOp(op, "+"); })
      .Case<arith::SubIOp>([&](auto op) { emitBinaryOp(op, "-"); })
      .Case<arith::MulIOp>([&](auto op) { emitBinaryOp(op, "*"); })
      .Case<arith::DivFOp>([&](auto op) { emitBinaryOp(op, "/"); })
      .Case<arith::DivUIOp>([&](auto op) { emitBinaryOp(op, "/"); })
      .Case<arith::DivSIOp>([&](auto op) { emitBinaryOp(op, "/"); })
      .Case<arith::RemSIOp>([&](auto op) { emitBinaryOp(op, "%"); })
      .Case<arith::RemUIOp>([&](auto op) { emitBinaryOp(op, "%"); })
      .Case<arith::AndIOp>([&](auto op) { emitBinaryOp(op, "&"); })
      .Case<arith::OrIOp>([&](auto op) { emitBinaryOp(op, "|"); })
      .Case<arith::XOrIOp>([&](auto op) { emitBinaryOp(op, "^"); })
      .Case<arith::ShLIOp>([&](auto op) { emitBinaryOp(op, "<<"); })
      .Case<arith::ShRUIOp>([&](auto op) { emitBinaryOp(op, ">>"); })
      .Case<arith::ShRSIOp>([&](auto op) { emitBinaryOp(op, ">>"); })
      .Case<arith::FloorDivSIOp>([&](auto op) { emitBinaryOp(op, "/"); })
      .Case<arith::CeilDivSIOp>([&](auto op) { emitBinaryOp(op, "/"); })
      .Case<arith::CeilDivUIOp>([&](auto op) { emitBinaryOp(op, "/"); })

      // max/min ops
      .Case<arith::MaxSIOp>([&](auto op) { emitMaxMinOp(op, "max"); })
      .Case<arith::MinSIOp>([&](auto op) { emitMaxMinOp(op, "min"); })
      .Case<arith::MaxUIOp>([&](auto op) { emitMaxMinOp(op, "max"); })
      .Case<arith::MinUIOp>([&](auto op) { emitMaxMinOp(op, "min"); })
      .Case<arith::MaximumFOp>([&](auto op) { emitMaxMinOp(op, "fmax"); })
      .Case<arith::MinimumFOp>([&](auto op) { emitMaxMinOp(op, "fmin"); })
      .Case<arith::MaxNumFOp>([&](auto op) { emitMaxMinOp(op, "fmax"); })
      .Case<arith::MinNumFOp>([&](auto op) { emitMaxMinOp(op, "fmin"); })

      // unary ops
      .Case<arith::NegFOp>([&](auto op) { emitUnaryOp(op, "-"); })
      .Case<math::AbsIOp>([&](auto op) { emitUnaryOp(op, "abs"); })
      .Case<math::AbsFOp>([&](auto op) { emitUnaryOp(op, "fabs"); })
      .Case<math::ExpOp>([&](auto op) { emitUnaryOp(op, "exp"); })
      .Case<math::Exp2Op>([&](auto op) { emitUnaryOp(op, "exp2"); })
      .Case<math::LogOp>([&](auto op) { emitUnaryOp(op, "log"); })
      .Case<math::Log2Op>([&](auto op) { emitUnaryOp(op, "log2"); })
      .Case<math::Log10Op>([&](auto op) { emitUnaryOp(op, "log10"); })
      .Case<math::SqrtOp>([&](auto op) { emitUnaryOp(op, "sqrt"); })
      .Case<math::RsqrtOp>([&](auto op) { emitUnaryOp(op, "1 / sqrt"); })
      .Case<math::SinOp>([&](auto op) { emitUnaryOp(op, "sin"); })
      .Case<math::CosOp>([&](auto op) { emitUnaryOp(op, "cos"); })
      .Case<math::TanOp>([&](auto op) { emitUnaryOp(op, "tan"); })
      .Case<math::SinhOp>([&](auto op) { emitUnaryOp(op, "sinh"); })
      .Case<math::CoshOp>([&](auto op) { emitUnaryOp(op, "cosh"); })
      .Case<math::TanhOp>([&](auto op) { emitUnaryOp(op, "tanh"); })
      .Case<math::PowFOp>([&](auto op) {
        state.os << "pow(";
        emitValue(op.getLhs());
        state.os << ", ";
        emitValue(op.getRhs());
        state.os << ");";
      })

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
  }
}

void VivadoHLSEmitter::emitBinaryOp(Operation *op,
                                    llvm::StringLiteral keyword) {
  llvm::raw_ostream &os = state.os;
  Value result = op->getResult(0);
  emitValue(result);
  os << " = ";
  emitValue(op->getOperand(0));
  os << " " << keyword << " ";
  emitValue(op->getOperand(1));
  os << ";";
}

void VivadoHLSEmitter::emitUnaryOp(Operation *op, llvm::StringLiteral keyword) {
  llvm::raw_ostream &os = state.os;
  Value result = op->getResult(0);
  emitValue(result);
  os << " = " << keyword << "(";
  emitValue(op->getOperand(0));
  os << ");";
}

void VivadoHLSEmitter::emitMaxMinOp(Operation *op,
                                    llvm::StringLiteral keyword) {
  llvm::raw_ostream &os = state.os;
  Value result = op->getResult(0);
  emitValue(result);
  os << " = " << keyword << "(";
  emitValue(op->getOperand(0));
  os << ", ";
  emitValue(op->getOperand(1));
  os << ");";
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
  emitValue(result);
  os << " = ";
  emitValue(op.getLhs());
  os << " " << getCmpIPredString(op.getPredicate()) << " ";
  emitValue(op.getRhs());
  os << ";";
}

void VivadoHLSEmitter::emitCmpF(arith::CmpFOp op) {
  llvm::raw_ostream &os = state.os;
  Value result = op.getResult();
  emitValue(result);
  os << " = ";
  emitValue(op.getLhs());
  os << " " << getCmpFPredString(op.getPredicate()) << " ";
  emitValue(op.getRhs());
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
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

constexpr llvm::StringLiteral hostHeader = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for host
//
//===----------------------------------------------------------------------===//
// standard C/C++ headers
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>

// vivado hls headers
#include "kernel.h"
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
)XXX";

void VivadoHLSEmitter::emitModule(ModuleOp mod) {
  // TODO: add host-side codegen
  llvm::raw_ostream &os = state.os;
  os << deviceHeader << "\n";
  // Step 1: emit global variables
  mod->walk([&](memref::GlobalOp op) { dispatch(op); });

  // Step 2: generate all function declarations
  for (auto func : mod.getOps<func::FuncOp>()) {
    emitFunctionReturnType(func);
    os << " " << func.getName() << "(";
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
  VivadoHLSEmitter emitter(os);
  emitter.state.indexWidth = indexWidth;
  emitter.state.indentSize = indent;
  emitter.state.withLocation = withLocation;
  emitter.emitModule(mod);
  return failure(emitter.state.failed);
}

void allo::registerVivadoHLSTranslation() {
  static TranslateFromMLIRRegistration reg(
      "emit-vitis-hls", "Translate MLIR to C++ code for Vivado HLS",
      emitVivadoHLS, [&](DialectRegistry &registry) {
        registry
            .insert<affine::AffineDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    func::FuncDialect, allo::AlloDialect>();
      });
}
