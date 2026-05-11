/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloTypes.h"

#include "allo/IR/AlloDialect.cpp.inc"

#include "allo/IR/AlloEnums.cpp.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.cpp.inc"

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
  SMLoc signatureLocation = p.getCurrentLocation();
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
  SMLoc attributeDictLocation = p.getCurrentLocation();
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

LogicalResult
GlobalStreamGetOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto globalStream = symbolTable.lookupNearestSymbolFrom<GlobalStreamCreateOp>(
      *this, getStreamAttr());
  if (!globalStream) {
    return emitOpError() << "referenced global stream '" << getStream()
                         << "' does not exist";
  }
  auto streamTy = globalStream.getStreamType();
  auto handleTy = cast<StreamType>(getHandle().getType());
  if (streamTy != handleTy) {
    return emitOpError() << "result type " << handleTy
                         << " does not match global stream type " << streamTy;
  }
  return success();
}
