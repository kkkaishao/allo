#include "allo/IR/AlloOps.h"
#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloTypes.h"

using namespace mlir;
using namespace mlir::allo;

#define GET_TYPEDEF_CLASSES
#include "allo/IR/AlloTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "allo/IR/AlloAttrs.cpp.inc"

#include "allo/IR/AlloEnums.cpp.inc"

#include "allo/IR/AlloInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.cpp.inc"

#include "allo/IR/AlloDialect.cpp.inc"

namespace {
// Used to customize partition attribute printing and parsing in the IR
struct AlloOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<PartitionAttr>(attr)) {
      os << "part";
      return AliasResult::OverridableAlias;
    }
    if (isa<VirtMapAttr>(attr)) {
      os << "virtmap";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void AlloDialect::initialize() {
  // clang-format off
  addTypes<
#define GET_TYPEDEF_LIST
#include "allo/IR/AlloTypes.cpp.inc"


      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "allo/IR/AlloAttrs.cpp.inc"


      >();
  addInterface<AlloOpAsmDialectInterface>();
  addOperations<
#define GET_OP_LIST
#include "allo/IR/AlloOps.cpp.inc"


      >();
  // clang-format on
}

LogicalResult
PartitionAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<PartitionKindEnum> kinds,
                      ArrayRef<int64_t> factors, ArrayRef<int64_t> dims) {
  if (kinds.size() != factors.size() || kinds.size() != dims.size()) {
    emitError()
        << "the size of 'kinds', 'factors', and 'dims' must be the same";
    return failure();
  }
  llvm::DenseSet<int64_t> seen;
  for (auto [dim, factor, kind] : llvm::zip_equal(dims, factors, kinds)) {
    if (dim < 0) {
      emitError() << "partition dimension must be non-negative";
      return failure();
    }
    if (seen.contains(dim)) {
      emitError() << "duplicate partition dimension: " << dim;
      return failure();
    }
    seen.insert(dim);
    if (kind != PartitionKindEnum::Complete) {
      if (factor <= 0) {
        emitError() << "partition factor must be a positive integer for "
                       "non-complete partition";
        return failure();
      }
      if (factor == 1) {
        emitError() << "partition factor must be greater than 1 for "
                       "non-complete partition";
        return failure();
      }
    }
    if (kind == PartitionKindEnum::Complete && factor != 0) {
      emitError() << "partition factor must be 0 for complete partition";
      return failure();
    }
  }
  return success();
}

void KernelOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs,
                     ArrayRef<int64_t> vtMap) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  VirtMapAttr vMapAttr;
  if (vtMap.empty()) {
    vMapAttr = VirtMapAttr::get(builder.getContext(), 1);
  } else {
    vMapAttr = VirtMapAttr::get(builder.getContext(), vtMap);
  }
  state.addAttribute(getVirtualMappingAttrName(state.name), vMapAttr);
  state.addAttributes(attrs);
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

//===----------------------------------------------------------------------===//
// Kernel Operations
//===----------------------------------------------------------------------===//

// KernelOp's parser and printer are adapted from func_interface_impl's parser
// and printer.
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/FunctionImplementation.cpp
void KernelOp::print(OpAsmPrinter &p) {
  auto funcName = getName();
  p << ' ';
  auto visibilityName = SymbolTable::getVisibilityAttrName();
  auto visibility = getSymVisibility();
  if (visibility.has_value()) {
    p << visibility.value() << ' ';
  }
  p.printSymbolName(funcName);
  // print virtual mapping
  auto virtmap = getVirtualMappingVec();
  if (virtmap.size() != 1 || virtmap.front() != 1) {
    p << " " << VirtMapAttrName << " ";
    p.printAttribute(getVirtualMappingAttr());
    p << " ";
  }
  ArrayRef<Type> argTypes = getArgumentTypes();
  ArrayRef<Type> resTypes = getResultTypes();
  function_interface_impl::printFunctionSignature(
      p, *this, argTypes, /*isVariadic=*/false, resTypes);
  function_interface_impl::printFunctionAttributes(
      p, *this,
      {visibilityName, getVirtualMappingAttrName(), getFunctionTypeAttrName(),
       getArgAttrsAttrName(), getResAttrsAttrName()});
  Region &body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse visibility.
  (void)impl::parseOptionalVisibilityKeyword(parser, result.attributes);
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();
  // Parse virtual mapping.
  if (succeeded(parser.parseOptionalKeyword(VirtMapAttrName))) {
    // virtual mapping is specified, parse the attribute.
    VirtMapAttr vMapAttr;
    if (parser.parseAttribute(vMapAttr, getVirtualMappingAttrName(result.name),
                              result.attributes))
      return failure();
  } else {
    // virtual mapping is not specified, use default value.
    result.addAttribute(getVirtualMappingAttrName(result.name),
                        VirtMapAttr::get(result.getContext(), 1));
  }
  // Parse the function signature.
  SMLoc signatureLocation = parser.getCurrentLocation();
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/false, entryArgs, isVariadic, resultTypes,
          resultAttrs))
    return failure();
  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  Type type = builder.getFunctionType(argTypes, resultTypes);
  if (!type) {
    return parser.emitError(signatureLocation)
           << "failed to construct function type";
  }
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();
  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed :
       {SymbolTable::getVisibilityAttrName(), SymbolTable::getSymbolAttrName(),
        getFunctionTypeAttrName(result.name).getValue()}) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);
  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resultAttrs, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  SMLoc loc = parser.getCurrentLocation();
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs,
                                 /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }
  return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = this->getProperties().callee;
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto fn = symbolTable.lookupNearestSymbolFrom<KernelOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

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
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

LogicalResult ReturnOp::verify() {
  auto function = cast<KernelOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

LogicalResult
VirtMapAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                    ArrayRef<int64_t> ranges) {
  if (llvm::any_of(ranges, [](int64_t x) { return x <= 0; })) {
    return emitError() << "ranges of virtual map must be all positive";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SPMD Operations
//===----------------------------------------------------------------------===//
LogicalResult GetProgramIdOp::verify() {
  auto parent = (*this)->getParentOfType<KernelOp>();
  if (!parent) {
    return emitOpError() << "must be nested inside a KernelOp";
  }
  auto dim = getAxiAttr().getInt();
  auto vMap = parent.getVirtualMapping();

  if (dim < 0 || static_cast<uint64_t>(dim) >= vMap.getRanges().size()) {
    return emitOpError()
           << "dimension " << dim
           << " is out of bounds for KernelOp virtual mapping of size "
           << vMap.getRanges().size();
  }
  return success();
}

LogicalResult GetNumProgramsOp::verify() {
  auto parent = (*this)->getParentOfType<KernelOp>();
  if (!parent) {
    return emitOpError() << "must be nested inside a KernelOp";
  }
  auto dim = getAxi();
  auto vMap = parent.getVirtualMapping();
  if (dim < 0 || static_cast<uint64_t>(dim) >= vMap.getRanges().size()) {
    return emitOpError()
           << "dimension " << dim
           << " is out of bounds for KernelOp virtual mapping of size "
           << vMap.getRanges().size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Stream Operations
//===----------------------------------------------------------------------===//

LogicalResult ChanPutOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto chan = SymbolTable::lookupNearestSymbolFrom<ChanCreateOp>(
      *this, getChannelAttr());
  if (!chan) {
    return emitOpError() << "must refer a valid channel";
  }
  if (getIndices().size() != chan.getShape().size()) {
    return emitOpError()
           << "rank of indices does not match that of the referred channel";
  }
  auto dataType = chan.getChanType().getDataType();
  auto valueTy = getValue().getType();
  if (dataType != valueTy) {
    return emitOpError() << "type mismatch between the input value and the "
                            "channel data type";
  }
  return success();
}

LogicalResult ChanGetOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto chan = SymbolTable::lookupNearestSymbolFrom<ChanCreateOp>(
      *this, getChannelAttr());
  if (!chan) {
    return emitOpError() << "must refer a valid channel";
  }
  if (getIndices().size() != chan.getShape().size()) {
    return emitOpError()
           << "rank of indices does not match that of the referred channel";
  }
  auto dataType = chan.getChanType().getDataType();
  auto valueTy = getValue().getType();
  if (dataType != valueTy) {
    return emitOpError() << "type mismatch between the return value and the "
                            "channel data type";
  }
  return success();
}

LogicalResult
ChanAcquireOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto chan = SymbolTable::lookupNearestSymbolFrom<ChanCreateOp>(
      *this, getChannelAttr());
  if (!chan) {
    return emitOpError() << "must refer a valid channel";
  }
  if (getIndices().size() != chan.getShape().size()) {
    return emitOpError()
           << "rank of indices does not match that of the channel";
  }
  auto dataType = chan.getChanType().getDataType();
  auto valueTys = getBuffers().getTypes();
  // check if value types are the same
  Type firstTy = valueTys.front();
  if (llvm::any_of(valueTys, [&](Type ty) { return ty != firstTy; })) {
    return emitOpError() << "return value types must be identical";
  }
  if (dataType != firstTy) {
    return emitOpError() << "type mismatch between the return value and the "
                            "channel data type";
  }
  int64_t size = getSizeAttr().getInt();
  if (size <= 0 ||
      static_cast<uint64_t>(size) > chan.getChanType().getDepth()) {
    return emitOpError() << "the number of buffers to acquire must be positive "
                            "and not exceed the depth of the channel";
  }
  return success();
}

LogicalResult
ChanReleaseOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto chan = SymbolTable::lookupNearestSymbolFrom<ChanCreateOp>(
      *this, getChannelAttr());
  if (!chan) {
    return emitOpError() << "must refer a valid channel";
  }
  if (getIndices().size() != chan.getShape().size()) {
    return emitOpError()
           << "rank of indices does not match that of the channel";
  }
  auto dataType = chan.getChanType().getDataType();
  auto valueTys = getBuffers().getTypes();
  // check if value types are the same
  Type firstTy = valueTys.front();
  if (llvm::any_of(valueTys, [&](Type ty) { return ty != firstTy; })) {
    return emitOpError() << "return value types must be identical";
  }
  if (dataType != firstTy) {
    return emitOpError() << "type mismatch between the return value and the "
                            "channel data type";
  }
  unsigned nBuffers = valueTys.size();
  if (nBuffers > chan.getChanType().getDepth()) {
    return emitOpError()
           << "the number of buffers to release must not exceed the "
              "depth of the channel";
  }
  return success();
}

LogicalResult BitExtractOp::verify() {
  auto resTy = getResult().getType();
  auto srcTy = getInput().getType();
  if (resTy.getWidth() > srcTy.getWidth()) {
    return emitOpError() << "requires the width of result must be less than or "
                            "equal to that of the input";
  }
  auto width = getWidthAttr().getInt();
  if (width <= 0 || width > srcTy.getWidth()) {
    return emitOpError()
           << "requires the width attribute must be a positive "
              "integer less than or equal to the width of the input";
  }
  return success();
}

LogicalResult BitInsertOp::verify() {
  auto resTy = getResult().getType();
  auto destTy = getDest().getType();
  if (resTy.getWidth() != destTy.getWidth()) {
    return emitOpError()
           << "requires the width of result must be equal to that "
              "of the input";
  }
  auto srcTy = getSrc().getType();
  if (srcTy.getWidth() > destTy.getWidth()) {
    return emitOpError() << "requires the width of input must be less than or "
                            "equal to that of the destination";
  }
  auto width = getWidthAttr().getInt();
  if (width <= 0 || width > destTy.getWidth()) {
    return emitOpError()
           << "requires the width attribute must be a positive "
              "integer less than or equal to the width of the input";
  }
  return success();
}

LogicalResult BitConcatOp::verify() {
  auto resWidth = getResult().getType().getWidth();
  int64_t totalWidth = 0;
  for (auto input : getInputs()) {
    auto inputWidth = cast<IntegerType>(input.getType()).getWidth();
    totalWidth += inputWidth;
  }
  if (resWidth != totalWidth) {
    return emitOpError() << "requires the width of result must be equal to the "
                            "sum of widths of all inputs";
  }
  return success();
}