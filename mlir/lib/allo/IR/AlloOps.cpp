/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloTypes.h"

// The generated ISA op parsers (custom<DynamicIndexList>) need these helpers.
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "allo/IR/AlloDialect.cpp.inc"

#include "allo/IR/AlloEnums.cpp.inc"

// ISA interfaces must precede the op/type classes that implement them.
#include "allo/IR/AlloOpInterfaces.cpp.inc"
#include "allo/IR/AlloOpsInterfaces.cpp.inc"
#include "allo/IR/AlloTypeInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.cpp.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloISAOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "allo/IR/AlloAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "allo/IR/AlloTypes.cpp.inc"

#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::allo;

LogicalResult
StreamType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                   Type baseType, std::size_t depth, ArrayRef<int64_t> shape) {
  if (!baseType)
    return emitError() << "expected stream base type";
  if (depth == 0)
    return emitError() << "stream depth must be positive";
  for (int64_t dim : shape) {
    if (dim < 0)
      return emitError() << "stream shape dimensions must be non-negative";
  }
  return success();
}

// ISA type/op method bodies live in AlloISATypes.cpp and AlloISAOps.cpp.

Type StreamType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  Type baseType;
  uint64_t depth = 0;
  SmallVector<int64_t> shape;
  if (parser.parseType(baseType) || parser.parseComma() ||
      parser.parseInteger(depth) || parser.parseComma() ||
      parser.parseLSquare())
    return {};

  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t dim = 0;
      if (parser.parseInteger(dim))
        return {};
      shape.push_back(dim);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return {};
  }

  if (parser.parseGreater())
    return {};
  return parser.getChecked<StreamType>(
      parser.getCurrentLocation(), parser.getContext(), baseType, depth, shape);
}

void StreamType::print(AsmPrinter &printer) const {
  printer << "<" << getBaseType() << "," << getDepth() << ",[";
  for (auto [idx, dim] : llvm::enumerate(getShape())) {
    if (idx != 0)
      printer << ",";
    printer << dim;
  }
  printer << "]>";
}

void KernelOp::print(OpAsmPrinter &p) {
  p << ' ';
  auto op = llvm::cast<FunctionOpInterface>(getOperation());
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibilty = op->getAttrOfType<StringAttr>(visibilityAttrName)) {
    p << visibilty.getValue() << ' ';
  }
  auto kName = getSymNameAttr().getValue();
  p.printSymbolName(kName);
  function_interface_impl::printFunctionSignature(p, op, getArgumentTypes(),
                                                  false, getResultTypes());
  p << " mapping=";
  p.printStrippedAttrOrType(getMappingAttr());
  function_interface_impl::printFunctionAttributes(
      p, op,
      {
          SymbolTable::getVisibilityAttrName(),
          getFunctionTypeAttrName(),
          getArgAttrsAttrName(),
          getMappingAttrName(),

      });
  Region &body = getRegion();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

ParseResult KernelOp::parse(OpAsmParser &p, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resAttrs;
  SmallVector<Type> resTypes;
  auto &builder = p.getBuilder();

  // Parse visibilty
  (void)impl::parseOptionalVisibilityKeyword(p, result.attributes);

  // Parse the name as a symbol
  StringAttr nameAttr;
  if (p.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                        result.attributes))
    return failure();

  // Parse the function signature
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          p, false, entryArgs, isVariadic, resTypes, resAttrs))
    return failure();
  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  FunctionType type = builder.getFunctionType(argTypes, resTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  // Parse the mapping attribute here
  if (p.parseKeyword("mapping") || p.parseEqual())
    return failure();
  DenseI32ArrayAttr mapping;
  if (p.parseCustomAttributeWithFallback(mapping, Type()))
    return failure();
  result.addAttribute(getMappingAttrName(result.name), mapping);

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  if (p.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  result.attributes.append(parsedAttributes);

  // Add the attributes to the function arguments.
  assert(resAttrs.size() == resTypes.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resAttrs, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  SMLoc loc = p.getCurrentLocation();
  OptionalParseResult parseResult =
      p.parseOptionalRegion(*body, entryArgs,
                            /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return p.emitError(loc, "expected non-empty function body");
  }
  return success();
}

void KernelOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     FunctionType type, ArrayRef<int32_t> mapping,
                     ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addAttribute(getMappingAttrName(state.name),
                     builder.getDenseI32ArrayAttr(mapping));
  state.attributes.append(attrs);
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

LogicalResult ReturnOp::verify() {
  auto kernel = cast<KernelOp>(this->getParentOp());
  auto results = kernel.getFunctionType().getResults();
  if (results.size() != getNumOperands())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << kernel.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in kernel @" << kernel.getSymName();
  return success();
}

LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  KernelOp fn = symbolTable.lookupNearestSymbolFrom<KernelOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid kernel";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "    op result types: " << getResultTypes();
      diag.attachNote() << "kernel result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

void AlloDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "allo/IR/AlloAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "allo/IR/AlloTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "allo/IR/AlloOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "allo/IR/AlloISAOps.cpp.inc"
      >();
}

LogicalResult
PartitionAxisAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          PartitionKindEnum kind, int64_t factor,
                          int64_t dims) {
  if (kind == PartitionKindEnum::CompletePartition && factor != 0) {
    return emitError() << "partition factor must be 0 for complete partition";
  }
  if (kind != PartitionKindEnum::CompletePartition && !(factor > 1)) {
    return emitError() << "partition factor must be greater than 1 for "
                          "non-complete partition";
  }
  if (dims < 0) {
    return emitError() << "dimension index must be non-negative";
  }
  return success();
}

LogicalResult
PartitionAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<PartitionAxisAttr> partitions) {
  DenseSet<int64_t> seen;
  for (auto &axi : partitions) {
    seen.insert(axi.getDim());
  }
  if (seen.size() < partitions.size()) {
    return emitError() << "duplicate partition axis detected";
  }
  return success();
}

LogicalResult StreamGetOp::verify() {
  auto streamTy = cast<StreamType>(getStream().getType());
  auto valueTy = getValue().getType();
  if (streamTy.getBaseType() != valueTy) {
    return emitOpError() << "stream type " << streamTy
                         << " does not match value type " << valueTy;
  }
  auto srcRank = streamTy.getShape().size();
  auto dstRank = getIndices().size();
  if (srcRank != dstRank) {
    return emitOpError() << "rank of stream (" << srcRank
                         << ") does not match number of indices (" << dstRank
                         << ")";
  }
  return success();
}

LogicalResult StreamPutOp::verify() {
  auto streamTy = cast<StreamType>(getStream().getType());
  auto valueTy = getValue().getType();
  if (streamTy.getBaseType() != valueTy) {
    return emitOpError() << "stream type " << streamTy
                         << " does not match value type " << valueTy;
  }
  auto dstRank = streamTy.getShape().size();
  auto srcRank = getIndices().size();
  if (srcRank != dstRank) {
    return emitOpError() << "rank of stream (" << dstRank
                         << ") does not match number of indices (" << srcRank
                         << ")";
  }
  return success();
}

LogicalResult AssumeNoDepOp::verify() {
  // A distance is an inter-iteration notion; an intra-iteration claim carries
  // none.
  if (getDistanceAttr() && getDepType() == AssumeDepTypeEnum::Intra)
    return emitOpError() << "'distance' is only meaningful for an inter-"
                            "iteration dependence (dep_type = inter)";
  return success();
}

//===----------------------------------------------------------------------===//
// Data & Control Path (dcp) operations
//===----------------------------------------------------------------------===//

namespace mlir::allo::dcp {

LogicalResult
DCPathUnitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (!symbolTable.lookupNearestSymbolFrom<DCPathOperatorOp>(*this,
                                                             getOpTypeAttr()))
    return emitOpError("references unknown operator type '")
           << getOpType() << "'";
  return success();
}

LogicalResult
DCPathComputeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (!symbolTable.lookupNearestSymbolFrom<DCPathOperatorOp>(*this,
                                                             getOpTypeAttr()))
    return emitOpError("references unknown operator type '")
           << getOpType() << "'";
  if (FlatSymbolRefAttr unit = getUnitAttr())
    if (!symbolTable.lookupNearestSymbolFrom<DCPathUnitOp>(*this, unit))
      return emitOpError("references unknown unit '") << unit.getValue() << "'";
  return success();
}

// A memory op's operator type is optional; verify it when present.
static LogicalResult verifyOptionalOperator(Operation *op,
                                            SymbolTableCollection &symbolTable,
                                            FlatSymbolRefAttr opr) {
  if (opr && !symbolTable.lookupNearestSymbolFrom<DCPathOperatorOp>(op, opr))
    return op->emitOpError("references unknown operator type '")
           << opr.getValue() << "'";
  return success();
}

LogicalResult
DCPathLoadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyOptionalOperator(*this, symbolTable, getOpTypeAttr());
}

LogicalResult
DCPathStoreOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyOptionalOperator(*this, symbolTable, getOpTypeAttr());
}

LogicalResult DCPathComputeOp::verify() {
  if (getStart() < 0)
    return emitOpError("start cycle must be non-negative");
  return success();
}

LogicalResult
DCPathInvokeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // The callee is a scheduled `func.func` at reify time (an `hw.module` after
  // emit); accept any symbol so the verifier survives both stages.
  if (!symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr()))
    return emitOpError("references unknown callee '") << getCallee() << "'";
  return success();
}

LogicalResult DCPathInvokeOp::verify() {
  if (getStart() < 0)
    return emitOpError("start cycle must be non-negative");
  return success();
}

LogicalResult DCPathPipelineOp::verify() {
  // `ii` is optional (absent for a data-dependent sequential wrapper); when
  // present it must be a positive initiation interval.
  if (std::optional<int64_t> ii = getIi(); ii && *ii < 1)
    return emitOpError("ii must be >= 1");
  if (std::optional<int64_t> s = getStep(); s && *s <= 0)
    return emitOpError("step must be > 0"); // A+ terminates on iv+step >= ub
  // A bound is either compile-time (attribute) or runtime (operand), never
  // both.
  if (getLbBound() && getLbAttr())
    return emitOpError("lb given as both an operand and an attribute");
  if (getStepBound() && getStepAttr())
    return emitOpError("step given as both an operand and an attribute");
  if (getLatencyBound() && !getLatencyAttr())
    return emitOpError("latency_bound requires latency");
  Block &body = getBody().front();
  if (body.getNumArguments() != 1 + getInits().size())
    return emitOpError(
        "body must have one induction argument plus one argument "
        "per iter-arg");
  if (!body.getArgument(0).getType().isIndex())
    return emitOpError(
        "the first body argument (induction variable) must have index type");

  // The terminator determines the loop kind: dcp.uncondition (counted) or
  // dcp.condition (while). Either carries one value per iter-arg; a while has
  // no trip (termination is its condition, not a counter).
  if (body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>())
    return emitOpError("body must end with a terminator");
  Operation *term = body.getTerminator();
  if (auto cond = dyn_cast<DCPathConditionOp>(term)) {
    if (getTripAttr())
      return emitOpError(
          "a while pipeline (dcp.condition terminator) must not have a trip");
    if (cond.getCarried().size() != getInits().size())
      return emitOpError("dcp.condition must carry one value per iter-arg");
  } else if (auto y = dyn_cast<DCPathUnconditionOp>(term)) {
    if (y.getOperands().size() != getInits().size())
      return emitOpError("dcp.uncondition must yield one value per iter-arg");
  } else {
    return emitOpError("body must end with dcp.uncondition or dcp.condition");
  }
  return success();
}

bool DCPathPipelineOp::isWhileLoop() {
  return isa<DCPathConditionOp>(getBody().front().getTerminator());
}

DCPathConditionOp DCPathPipelineOp::getConditionOp() {
  return dyn_cast<DCPathConditionOp>(getBody().front().getTerminator());
}

DCPathUnconditionOp DCPathPipelineOp::getUnconditionOp() {
  return dyn_cast<DCPathUnconditionOp>(getBody().front().getTerminator());
}

Value DCPathPipelineOp::getConditionValue() {
  DCPathConditionOp c = getConditionOp();
  return c ? c.getCondition() : Value();
}

OperandRange DCPathPipelineOp::getCarriedValues() {
  if (DCPathConditionOp c = getConditionOp())
    return c.getCarried();
  return getUnconditionOp().getOperands();
}

//===----------------------------------------------------------------------===//
// dcp.pipeline / dcp.sequential custom assembly
//===----------------------------------------------------------------------===//

void DCPathPipelineOp::print(OpAsmPrinter &p) {
  Block &body = getBody().front();
  int64_t lb = getLb().value_or(0), step = getStep().value_or(1);
  p << ' ' << body.getArgument(0) << " = ";
  if (Value l = getLbBound())
    p << l; // a runtime lower bound (data-dependent range start)
  else
    p << lb;
  p << " to ";
  if (std::optional<int64_t> t = getTrip())
    p << (lb + *t * step); // the derived upper bound (ub = lb + trip*step)
  else if (Value b = getDynamicBound())
    p << b; // a runtime upper bound (dynamic trip)
  else
    p << '?'; // a while loop (termination by dcp.condition)
  if (Value s = getStepBound())
    p << " step " << s; // a runtime stride
  else if (step != 1)
    p << " step " << step;
  if (IntegerAttr ii = getIiAttr())
    p << " ii=" << ii.getInt();
  if (IntegerAttr s = getStartAttr())
    p << " at " << s.getInt();
  if (IntegerAttr l = getLengthAttr())
    p << " length=" << l.getInt();
  if (IntegerAttr lat = getLatencyAttr()) {
    p << " lat=" << lat.getInt();
    if (getLatencyBound())
      p << " bound";
  }
  if (!getInits().empty()) {
    p << " iter_args(";
    for (unsigned i = 0, e = getInits().size(); i < e; ++i) {
      if (i)
        p << ", ";
      p << body.getArgument(i + 1) << " = " << getInits()[i];
    }
    p << ")";
  }
  if (getNumResults()) {
    p << " -> (";
    for (unsigned i = 0, e = getNumResults(); i < e; ++i) {
      if (i)
        p << ", ";
      p << getResult(i).getType();
    }
    p << ")";
  }
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getTripAttrName(), getLbAttrName(), getStepAttrName(),
                       getIiAttrName(), getStartAttrName(), getLengthAttrName(),
                       getLatencyAttrName(), getLatencyBoundAttrName(),
                       getOperandSegmentSizesAttrName()});
}

ParseResult DCPathPipelineOp::parse(OpAsmParser &p, OperationState &result) {
  Builder &b = p.getBuilder();
  OpAsmParser::Argument iv;
  iv.type = b.getIndexType();
  int64_t lb = 0, ii;
  if (p.parseArgument(iv) || p.parseEqual())
    return failure();
  // Lower bound after `=`: an SSA `%operand` (a runtime lb -- the `lbBound`
  // operand, resolved first so it leads the operand segments, matching the
  // declared order lbBound, dynamicBound, stepBound) or an integer (a
  // compile-time `lb`, default 0).
  bool hasLb = false;
  {
    OpAsmParser::UnresolvedOperand lbOp;
    OptionalParseResult res = p.parseOptionalOperand(lbOp);
    if (res.has_value()) {
      if (failed(*res) ||
          p.resolveOperand(lbOp, b.getIndexType(), result.operands))
        return failure();
      hasLb = true;
    } else if (p.parseInteger(lb)) {
      return failure();
    }
  }
  if (p.parseKeyword("to"))
    return failure();
  // Termination bound after `to`, three forms: `?` (a while loop, terminated by
  // dcp.condition -- no trip, no bound); an SSA `%operand` (a runtime upper
  // bound -- the `dynamicBound` operand); or an integer (a compile-time upper
  // bound `ub`, from which the derived `trip` count is computed below).
  bool hasBound = false, hasUb = false;
  int64_t ub = 0;
  if (succeeded(p.parseOptionalQuestion())) {
    // while loop: leave trip / dynamicBound / lb / step unset
  } else {
    OpAsmParser::UnresolvedOperand boundOp;
    OptionalParseResult res = p.parseOptionalOperand(boundOp);
    if (res.has_value()) {
      if (failed(*res) ||
          p.resolveOperand(boundOp, b.getIndexType(), result.operands))
        return failure();
      hasBound = true; // resolved first, so it precedes inits in the segments
    } else {
      if (p.parseInteger(ub))
        return failure();
      hasUb = true;
    }
  }
  // Optional `step` (default 1): an SSA `%operand` (a runtime `stepBound`) or
  // an integer. Recorded (with `lb`) as an attribute only when a compile-time
  // non-default, so the common `lb=0`/`step=1` form round-trips to today's
  // syntax.
  int64_t step = 1;
  bool hasStep = false;
  if (succeeded(p.parseOptionalKeyword("step"))) {
    OpAsmParser::UnresolvedOperand stepOp;
    OptionalParseResult res = p.parseOptionalOperand(stepOp);
    if (res.has_value()) {
      if (failed(*res) ||
          p.resolveOperand(stepOp, b.getIndexType(), result.operands))
        return failure();
      hasStep = true;
    } else if (p.parseInteger(step)) {
      return failure();
    }
  }
  if (!hasLb && lb != 0)
    result.addAttribute(getLbAttrName(result.name), b.getI64IntegerAttr(lb));
  if (!hasStep && step != 1)
    result.addAttribute(getStepAttrName(result.name),
                        b.getI64IntegerAttr(step));
  // The constant iteration count is derived from the range ceil((ub-lb)/step)
  // only when every bound is compile-time (a runtime lb/step has no static
  // trip).
  if (hasUb && !hasLb && !hasStep)
    result.addAttribute(getTripAttrName(result.name),
                        b.getI64IntegerAttr(std::max<int64_t>(
                            0, llvm::divideCeilSigned(ub - lb, step))));
  // `ii` is optional: absent for a data-dependent sequential wrapper.
  if (succeeded(p.parseOptionalKeyword("ii"))) {
    if (p.parseEqual() || p.parseInteger(ii))
      return failure();
    result.addAttribute(getIiAttrName(result.name), b.getI64IntegerAttr(ii));
  }
  if (succeeded(p.parseOptionalKeyword("at"))) {
    int64_t start;
    if (p.parseInteger(start))
      return failure();
    result.addAttribute(getStartAttrName(result.name),
                        b.getI64IntegerAttr(start));
  }
  if (succeeded(p.parseOptionalKeyword("length"))) {
    int64_t length;
    if (p.parseEqual() || p.parseInteger(length))
      return failure();
    result.addAttribute(getLengthAttrName(result.name),
                        b.getI64IntegerAttr(length));
  }
  if (succeeded(p.parseOptionalKeyword("lat"))) {
    int64_t latency;
    if (p.parseEqual() || p.parseInteger(latency))
      return failure();
    result.addAttribute("latency", b.getI64IntegerAttr(latency));
    if (succeeded(p.parseOptionalKeyword("bound")))
      result.addAttribute(getLatencyBoundAttrName(result.name),
                          b.getUnitAttr());
  }

  SmallVector<OpAsmParser::Argument> regionArgs{iv};
  SmallVector<OpAsmParser::UnresolvedOperand> inits;
  if (succeeded(p.parseOptionalKeyword("iter_args"))) {
    SmallVector<OpAsmParser::Argument> iterArgs;
    if (p.parseAssignmentList(iterArgs, inits))
      return failure();
    regionArgs.append(iterArgs.begin(), iterArgs.end());
  }

  SmallVector<Type> resultTypes;
  if (succeeded(p.parseOptionalArrow()))
    if (p.parseLParen() || p.parseTypeList(resultTypes) || p.parseRParen())
      return failure();
  if (resultTypes.size() != inits.size())
    return p.emitError(p.getNameLoc(), "expected one result type per iter-arg");
  result.addTypes(resultTypes);
  for (unsigned i = 0, e = inits.size(); i < e; ++i)
    regionArgs[i + 1].type = resultTypes[i];
  if (p.resolveOperands(inits, resultTypes, p.getCurrentLocation(),
                        result.operands))
    return failure();
  // AttrSizedOperandSegments: the three optional bound operands (lbBound,
  // dynamicBound, stepBound -- each 0 or 1) precede the inits in
  // result.operands, resolved above in that declared order.
  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      b.getDenseI32ArrayAttr({hasLb ? 1 : 0, hasBound ? 1 : 0, hasStep ? 1 : 0,
                              static_cast<int32_t>(inits.size())}));

  Region *region = result.addRegion();
  if (p.parseRegion(*region, regionArgs) ||
      p.parseOptionalAttrDict(result.attributes))
    return failure();
  // Default to an unconditional terminator when the body has none; a while
  // pipeline prints its dcp.condition explicitly (the terminator is no longer
  // implicit, so this replaces the SingleBlockImplicitTerminator hook).
  Block &blk = region->front();
  if (blk.empty() || !blk.back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder tb = OpBuilder::atBlockEnd(&blk);
    DCPathUnconditionOp::create(tb, result.location);
  }
  return success();
}

LogicalResult DCPathSequentialOp::verify() {
  if (getLatencyBound() && !getLatencyAttr())
    return emitOpError("latency_bound requires latency");
  return success();
}

void DCPathSequentialOp::print(OpAsmPrinter &p) {
  if (IntegerAttr s = getStartAttr())
    p << " at " << s.getInt();
  if (IntegerAttr l = getLengthAttr())
    p << " length=" << l.getInt();
  if (IntegerAttr lat = getLatencyAttr()) {
    p << " lat=" << lat.getInt();
    if (getLatencyBound())
      p << " bound";
  }
  if (getNumResults()) {
    p << " -> (";
    for (unsigned i = 0, e = getNumResults(); i < e; ++i) {
      if (i)
        p << ", ";
      p << getResult(i).getType();
    }
    p << ")";
  }
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getStartAttrName(), getLengthAttrName(),
                       getLatencyAttrName(), getLatencyBoundAttrName()});
}

ParseResult DCPathSequentialOp::parse(OpAsmParser &p, OperationState &result) {
  Builder &b = p.getBuilder();
  if (succeeded(p.parseOptionalKeyword("at"))) {
    int64_t start;
    if (p.parseInteger(start))
      return failure();
    result.addAttribute(getStartAttrName(result.name),
                        b.getI64IntegerAttr(start));
  }
  if (succeeded(p.parseOptionalKeyword("length"))) {
    int64_t length;
    if (p.parseEqual() || p.parseInteger(length))
      return failure();
    result.addAttribute(getLengthAttrName(result.name),
                        b.getI64IntegerAttr(length));
  }
  if (succeeded(p.parseOptionalKeyword("lat"))) {
    int64_t latency;
    if (p.parseEqual() || p.parseInteger(latency))
      return failure();
    result.addAttribute("latency", b.getI64IntegerAttr(latency));
    if (succeeded(p.parseOptionalKeyword("bound")))
      result.addAttribute(getLatencyBoundAttrName(result.name),
                          b.getUnitAttr());
  }
  SmallVector<Type> resultTypes;
  if (succeeded(p.parseOptionalArrow()))
    if (p.parseLParen() || p.parseTypeList(resultTypes) || p.parseRParen())
      return failure();
  result.addTypes(resultTypes);
  Region *region = result.addRegion();
  if (p.parseRegion(*region) || p.parseOptionalAttrDict(result.attributes))
    return failure();
  ensureTerminator(*region, b, result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// dcp.select custom assembly
//===----------------------------------------------------------------------===//

// One branch of a dcp.select must end with a dcp.uncondition yielding one value
// per select result. \p required rejects an empty branch (the then branch, and
// the else branch when there are results -- a mux needs both sources).
static LogicalResult verifySelectBranch(DCPathSelectOp op, Region &r,
                                        bool required, StringRef which) {
  if (r.empty()) {
    if (required)
      return op.emitOpError() << which << " branch must be present";
    return success();
  }
  Block &blk = r.front();
  if (blk.empty() || !blk.back().hasTrait<OpTrait::IsTerminator>())
    return op.emitOpError() << which << " branch must end with a terminator";
  auto term = dyn_cast<DCPathUnconditionOp>(blk.getTerminator());
  if (!term)
    return op.emitOpError() << which << " branch must end with dcp.uncondition";
  if (term.getOperands().size() != op.getNumResults())
    return op.emitOpError()
           << which << " branch must yield one value per select result";
  return success();
}

LogicalResult DCPathSelectOp::verify() {
  if (getLatencyBound() && !getLatencyAttr())
    return emitOpError("latency_bound requires latency");
  if (failed(verifySelectBranch(*this, getThenRegion(), /*required=*/true,
                                "then")))
    return failure();
  // The else branch is required exactly when results are yielded (the derived
  // result-mux needs a value from both paths); otherwise it is optional.
  return verifySelectBranch(*this, getElseRegion(),
                            /*required=*/getNumResults() > 0, "else");
}

void DCPathSelectOp::print(OpAsmPrinter &p) {
  p << ' ' << getCondition();
  if (IntegerAttr s = getStartAttr())
    p << " at " << s.getInt();
  if (IntegerAttr lat = getLatencyAttr()) {
    p << " lat=" << lat.getInt();
    if (getLatencyBound())
      p << " bound";
  }
  if (getNumResults()) {
    p << " -> (";
    for (unsigned i = 0, e = getNumResults(); i < e; ++i) {
      if (i)
        p << ", ";
      p << getResult(i).getType();
    }
    p << ")";
  }
  p << ' ';
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  if (!getElseRegion().empty()) {
    p << " else ";
    p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getStartAttrName(),
                                           getLatencyAttrName(),
                                           getLatencyBoundAttrName()});
}

ParseResult DCPathSelectOp::parse(OpAsmParser &p, OperationState &result) {
  Builder &b = p.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  if (p.parseOperand(cond))
    return failure();
  if (succeeded(p.parseOptionalKeyword("at"))) {
    int64_t start;
    if (p.parseInteger(start))
      return failure();
    result.addAttribute(getStartAttrName(result.name),
                        b.getI64IntegerAttr(start));
  }
  if (succeeded(p.parseOptionalKeyword("lat"))) {
    int64_t latency;
    if (p.parseEqual() || p.parseInteger(latency))
      return failure();
    result.addAttribute("latency", b.getI64IntegerAttr(latency));
    if (succeeded(p.parseOptionalKeyword("bound")))
      result.addAttribute(getLatencyBoundAttrName(result.name),
                          b.getUnitAttr());
  }
  SmallVector<Type> resultTypes;
  if (succeeded(p.parseOptionalArrow()))
    if (p.parseLParen() || p.parseTypeList(resultTypes) || p.parseRParen())
      return failure();
  if (p.resolveOperand(cond, b.getI1Type(), result.operands))
    return failure();
  result.addTypes(resultTypes);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  if (p.parseRegion(*thenRegion))
    return failure();
  if (succeeded(p.parseOptionalKeyword("else")))
    if (p.parseRegion(*elseRegion))
      return failure();
  return p.parseOptionalAttrDict(result.attributes);
}

} // namespace mlir::allo::dcp
