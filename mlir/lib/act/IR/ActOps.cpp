#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "act/IR/ActOps.h"

#include "act/IR/ActDialect.cpp.inc"

#include "act/IR/ActOpInterfaces.cpp.inc"

#include "act/IR/ActTypesInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "act/IR/ActTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "act/IR/ActAttrs.cpp.inc"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::act;

#define GET_OP_CLASSES
#include "act/IR/ActOps.cpp.inc"

void ActDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "act/IR/ActAttrs.cpp.inc"

      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "act/IR/ActTypes.cpp.inc"

      >();
  addOperations<
#define GET_OP_LIST
#include "act/IR/ActOps.cpp.inc"

      >();
}

LogicalResult DeclareBufferOp::verify() {
  auto bufferType = getBufferType();
  if (isa<HBMBufferType>(bufferType) && getSize() != 1)
    return emitError() << "global HBM buffers must have size 1";
  return success();
}

LogicalResult DeclareStateOp::verify() {
  auto defaultOr = getDefaultState();
  if (!defaultOr)
    return success();
  auto enums = getEnums().getAsRange<StringAttr>();
  if (llvm::find(enums, StringAttr::get(getContext(), *defaultOr)) ==
      enums.end())
    return emitError() << "default state must be one of the enumerated states";
  return success();
}

LogicalResult
WriteStateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto state = symbolTable.lookupNearestSymbolFrom<DeclareStateOp>(
      *this, getStateAttr());
  if (!state)
    return emitError() << "referred state '" << getState()
                       << "' does not exist";
  auto enums = state.getEnums().getAsRange<StringAttr>();
  if (llvm::find(enums, getValueAttr()) == enums.end())
    return emitError() << "value must be one of the enumerated states";
  return success();
}

LogicalResult
ReadStateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto state = symbolTable.lookupNearestSymbolFrom<DeclareStateOp>(
      *this, getStateAttr());
  if (!state)
    return emitError() << "referred state '" << getState()
                       << "' does not exist";
  auto eltType = state.getStateType().getElementType();
  if (eltType != getType())
    return emitError() << "expected type '" << eltType << "' but got '"
                       << getType() << "'";
  return success();
}

LogicalResult
WriteDescFieldOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto desc =
      symbolTable.lookupNearestSymbolFrom<DeclareDescOp>(*this, getDescAttr());
  if (!desc)
    return emitError() << "referred descriptor '" << getDesc()
                       << "' does not exist";
  DescriptorType descType = desc.getDescType();
  auto fields = descType.getFields();
  auto *it = llvm::find(fields, getFieldAttr());
  if (it == fields.end())
    return emitError() << "field must be one of the descriptor fields";
  auto fieldType = descType.getFieldTypes()[std::distance(fields.begin(), it)];
  if (fieldType != getValue().getType())
    return emitError() << "expected type '" << fieldType << "' but got '"
                       << getValue().getType() << "'";
  return success();
}

LogicalResult
ReadDescFieldOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto desc =
      symbolTable.lookupNearestSymbolFrom<DeclareDescOp>(*this, getDescAttr());
  if (!desc)
    return emitError() << "referred descriptor '" << getDesc()
                       << "' does not exist";
  DescriptorType descType = desc.getDescType();
  auto fields = descType.getFields();
  auto *it = llvm::find(fields, getFieldAttr());
  if (it == fields.end())
    return emitError() << "field must be one of the descriptor fields";
  auto fieldType = descType.getFieldTypes()[std::distance(fields.begin(), it)];
  if (fieldType != getType())
    return emitError() << "expected type '" << fieldType << "' but got '"
                       << getType() << "'";
  return success();
}

LogicalResult StridedOp::verify() {
  if (getStaticStrides().size() != getStaticCounts().size() ||
      getStaticStrides().size() != getStaticBasis().size())
    return emitError()
           << "basis, counts and strides must have the same number of "
              "elements";
  for (auto stride : getStaticStrides()) {
    if (ShapedType::isStatic(stride) && stride <= 0)
      // stride == 0 may be useful for broadcasting, but we disallow it for
      // simplicity
      return emitError() << "strides must be positive";
  }
  for (auto count : getStaticCounts()) {
    if (ShapedType::isStatic(count) && count <= 0)
      return emitError() << "counts must be positive";
  }
  for (auto base : getStaticBasis()) {
    if (ShapedType::isStatic(base) && base < 0)
      return emitError() << "basis must be non-negative";
  }
  return success();
}

LogicalResult StridedOp::verifyCompatibility(BufferTypeInterface bufferType,
                                             unsigned size) {
  unsigned dims = getStaticCounts().size();
  // Rule 1: dimensionality check
  if (!isa<HBMBufferType>(bufferType) && dims != 1)
    return emitError() << "on-chip buffers must be accessed in 1D patterns";
  // Rule 2: bounds check (static only)
  auto hbm = dyn_cast<HBMBufferType>(bufferType);
  ArrayRef<int64_t> shape = hbm ? hbm.getShape() : size;
  auto basis = getStaticBasis();
  auto strides = getStaticStrides();
  auto counts = getStaticCounts();
  for (unsigned i = 0; i < dims; ++i) {
    if (ShapedType::isStatic(strides[i]) && ShapedType::isStatic(counts[i])) {
      int64_t maxIndex = strides[i] * (counts[i] - 1);
      if (ShapedType::isStatic(basis[i]))
        maxIndex += basis[i];
      if (maxIndex >= shape[i])
        return emitError() << "access out of bounds in dimension " << i
                           << ": max index is " << maxIndex
                           << " but dimension size is " << shape[i];
    }
  }
  return success();
}

Value StridedOp::materialize(OpBuilder &builder, Location loc, Value buffer,
                             bool enableTensor) {
  MLIRContext *ctx = builder.getContext();
  auto mixedBasis = getMixedValues(getStaticBasis(), getBasis(), ctx);
  auto mixedStrides = getMixedValues(getStaticStrides(), getStrides(), ctx);
  auto mixedCounts = getMixedValues(getStaticCounts(), getCounts(), ctx);
  auto shaped = cast<ShapedType>(buffer.getType());
  unsigned currRank = mixedBasis.size();
  // align the dimensions
  assert(shaped.getRank() >= currRank);
  unsigned diff = shaped.getRank() - currRank;
  mixedBasis.append(diff, builder.getI64IntegerAttr(0));
  mixedStrides.append(diff, builder.getI64IntegerAttr(1));
  for (int i = 0; i < diff; ++i)
    mixedCounts.push_back(
        builder.getI64IntegerAttr(shaped.getDimSize(i + currRank)));
  Value extracted;
  if (enableTensor) {
    // drop leading dimension if equals to 1
    auto resultTy = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
        shaped.getRank() - 1, cast<RankedTensorType>(shaped), mixedCounts);
    extracted = tensor::ExtractSliceOp::create(
        builder, loc, resultTy, buffer, mixedBasis, mixedCounts, mixedStrides);
  } else {
    auto resultTy = memref::SubViewOp::inferRankReducedResultType(
        shaped.getRank() - 1, cast<MemRefType>(shaped), mixedBasis, mixedCounts,
        mixedStrides);
    extracted = memref::SubViewOp::create(
        builder, loc, resultTy, buffer, mixedBasis, mixedCounts, mixedStrides);
  }
  return extracted;
}

Value StridedOp::materialize(OpBuilder &builder, Location loc, Value value,
                             Value buffer) {
  MLIRContext *ctx = builder.getContext();
  auto mixedBasis = getMixedValues(getStaticBasis(), getBasis(), ctx);
  auto mixedStrides = getMixedValues(getStaticStrides(), getStrides(), ctx);
  auto mixedCounts = getMixedValues(getStaticCounts(), getCounts(), ctx);
  auto shaped = cast<ShapedType>(buffer.getType());
  unsigned currRank = mixedBasis.size();
  assert(shaped.getRank() >= currRank);
  unsigned diff = shaped.getRank() - currRank;
  // align the dimensions
  mixedBasis.append(diff, builder.getI64IntegerAttr(0));
  mixedStrides.append(diff, builder.getI64IntegerAttr(1));
  for (int i = 0; i < diff; ++i)
    mixedCounts.push_back(
        builder.getI64IntegerAttr(shaped.getDimSize(i + currRank)));
  Value inserted = tensor::InsertSliceOp::create(
      builder, loc, value, buffer, mixedBasis, mixedCounts, mixedStrides);
  return inserted;
}

LogicalResult TiledOp::verify() {
  // Same dimension consistency check as StridedOp
  if (getStaticStrides().size() != getStaticCounts().size() ||
      getStaticStrides().size() != getStaticBasis().size() ||
      getStaticStrides().size() != getStaticTileSizes().size())
    return emitError() << "basis, counts, strides and tile_sizes must "
                          "have the same number of elements";

  for (auto stride : getStaticStrides())
    if (ShapedType::isStatic(stride) && stride <= 0)
      return emitError() << "strides must be positive";

  for (auto count : getStaticCounts())
    if (ShapedType::isStatic(count) && count <= 0)
      return emitError() << "counts must be positive";

  for (auto base : getStaticBasis())
    if (ShapedType::isStatic(base) && base < 0)
      return emitError() << "basis must be non-negative";

  for (auto tileSize : getStaticTileSizes())
    if (ShapedType::isStatic(tileSize) && tileSize <= 0)
      return emitError() << "tile_sizes must be positive";

  // Tile size must not exceed stride — otherwise tiles overlap
  // which is undefined behavior on most hardware
  auto strides = getStaticStrides();
  auto tileSizes = getStaticTileSizes();
  for (unsigned i = 0; i < strides.size(); ++i)
    if (ShapedType::isStatic(strides[i]) &&
        ShapedType::isStatic(tileSizes[i]) && tileSizes[i] > strides[i])
      return emitError() << "tile_size " << tileSizes[i] << " exceeds stride "
                         << strides[i] << " in dimension " << i
                         << ": tiles would overlap";

  return success();
}

LogicalResult TiledOp::verifyCompatibility(BufferTypeInterface bufferType,
                                           unsigned size) {
  unsigned dims = getStaticCounts().size();

  // Rule 1: dimensionality — same as StridedOp
  if (!isa<HBMBufferType>(bufferType) && dims != 1)
    return emitError() << "on-chip buffers must be accessed in 1D patterns";

  // Rule 2: tile size must divide buffer element size
  // Only applies to on-chip buffers — HBM has no element granularity
  if (!isa<HBMBufferType>(bufferType)) {
    auto tileSizes = getStaticTileSizes();
    int64_t elemSize = std::accumulate(bufferType.getShape().begin(),
                                       bufferType.getShape().end(), 1,
                                       std::multiplies<int64_t>());
    // for scalar: elemSize = 1
    // for vector<NxE>: elemSize = N
    // for matrix<NxMxE>: elemSize = N*M (or per-dim check)
    if (ShapedType::isStatic(tileSizes[0])) {
      if (elemSize % tileSizes[0] != 0)
        return emitError() << "tile_size " << tileSizes[0]
                           << " does not evenly divide buffer element size "
                           << elemSize;
    }
  }

  // Rule 3: bounds check
  // Key difference from StridedOp: footprint includes tile extent
  // max_index = basis + stride * (count - 1) + (tile_size - 1)
  auto hbm = dyn_cast<HBMBufferType>(bufferType);
  ArrayRef<int64_t> shape = hbm ? hbm.getShape() : size;
  auto basis = getStaticBasis();
  auto strides = getStaticStrides();
  auto counts = getStaticCounts();
  auto tileSizes = getStaticTileSizes();

  for (unsigned i = 0; i < dims; ++i) {
    if (ShapedType::isStatic(strides[i]) && ShapedType::isStatic(counts[i]) &&
        ShapedType::isStatic(tileSizes[i])) {
      // Footprint without base
      int64_t maxIndex = strides[i] * (counts[i] - 1) + (tileSizes[i] - 1);
      if (ShapedType::isStatic(basis[i]))
        maxIndex += basis[i];
      if (maxIndex >= shape[i])
        return emitError() << "tiled access out of bounds in dimension " << i
                           << ": max index is " << maxIndex
                           << " but dimension size is " << shape[i];
    }
  }
  return success();
}

Value TiledOp::materialize(OpBuilder &builder, Location loc, Value buffer,
                           bool enableTensor) {
  return {}; // TODO
}

Value TiledOp::materialize(OpBuilder &builder, Location loc, Value value,
                           Value buffer) {
  return {}; // TODO
}

void DefineOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << " {";
  p.increaseIndent();
  p.printNewline();
  p << "src(";
  llvm::interleaveComma(
      getSources().getAsRange<FlatSymbolRefAttr>(), p,
      [&](FlatSymbolRefAttr src) { p.printSymbolName(src.getValue()); });
  p << ") dst(";
  llvm::interleaveComma(
      getDestinations().getAsRange<FlatSymbolRefAttr>(), p,
      [&](FlatSymbolRefAttr dst) { p.printSymbolName(dst.getValue()); });
  p << ")";
  p.printNewline();
  p << "addr(";
  llvm::interleaveComma(getAccessBlock().getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ") ";
  p.printRegion(getAccess(), /*printEntryBlockArgs=*/false);
  p.printNewline();
  p << "compute(";
  llvm::interleaveComma(getSemanticsBlock().getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ")";
  p.printRegion(getSemantics(), /*printEntryBlockArgs=*/false);
  p.decreaseIndent();
  p.printNewline();
  p << '}';
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{SymbolTable::getSymbolAttrName(),
                                           getSourcesAttrName(),
                                           getDestinationsAttrName()});
}

ParseResult DefineOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symName;
  SmallVector<Attribute, 4> srcs, dsts;
  if (parser.parseSymbolName(symName) || parser.parseLBrace() ||
      parser.parseKeyword("src") || parser.parseLParen() ||
      parser.parseCommaSeparatedList([&]() {
        FlatSymbolRefAttr src;
        if (parser.parseAttribute(src))
          return failure();
        srcs.push_back(src);
        return success();
      }) ||
      parser.parseRParen() || parser.parseKeyword("dst") ||
      parser.parseLParen() || parser.parseCommaSeparatedList([&]() {
        FlatSymbolRefAttr dst;
        if (parser.parseAttribute(dst))
          return failure();
        dsts.push_back(dst);
        return success();
      }) ||
      parser.parseRParen())
    return failure();
  auto builder = parser.getBuilder();
  result.addAttribute(SymbolTable::getSymbolAttrName(), symName);
  result.addAttribute(getSourcesAttrName(result.name),
                      builder.getArrayAttr(srcs));
  result.addAttribute(getDestinationsAttrName(result.name),
                      builder.getArrayAttr(dsts));
  Region *accessRegion = result.addRegion();
  Region *semanticsRegion = result.addRegion();
  SmallVector<OpAsmParser::Argument, 4> addrArgs, computeArgs;
  if (parser.parseKeyword("addr") ||
      parser.parseArgumentList(addrArgs, AsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();
  // parse access region
  if (parser.parseRegion(*accessRegion, addrArgs))
    return failure();
  if (parser.parseKeyword("compute") ||
      parser.parseArgumentList(computeArgs, AsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();
  // parse semantics region
  if (parser.parseRegion(*semanticsRegion, computeArgs))
    return failure();
  if (parser.parseRBrace())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

LogicalResult DefineOp::verify() {
  Block &access = getAccessBlock();
  if (access.empty() || !isa<YieldOp>(access.back()))
    return emitError() << "access region must end with a yield op";
  Operation &accYield = access.back();
  unsigned nBuffers = getSources().size() + getDestinations().size();
  if (accYield.getNumOperands() != nBuffers)
    return emitError()
           << "access region must yield the same number of buffer access "
              "patterns as the number of source and destination buffers";
  if (llvm::any_of(accYield.getOperands(), [](Value v) {
        return v.getDefiningOp<BufferAccessOpInterface>() == nullptr;
      }))
    return emitError()
           << "access region must yield only buffer access patterns";

  if (llvm::any_of(access.getArgumentTypes(),
                   [](Type t) { return !t.isIndex(); }))
    return emitError() << "access region arguments must all be index type";

  Block &semantics = getSemanticsBlock();
  if (semantics.empty() || !isa<YieldOp>(semantics.back()))
    return emitError() << "semantics region must end with a yield op";
  Operation &semYield = semantics.back();
  if (semYield.getNumOperands() < getDestinations().size())
    return emitError() << "semantics region must yield the same number of "
                          "values as the number of destination buffers";
  if (semantics.getNumArguments() < nBuffers)
    return emitError()
           << "number of semantics region arguments must be at least the total "
              "number of source and destination buffers";

  auto bufferTys = llvm::drop_end(semantics.getArgumentTypes(),
                                  semantics.getNumArguments() - nBuffers);
  auto computeParamTys =
      llvm::drop_begin(semantics.getArgumentTypes(), nBuffers);
  for (auto argTy : bufferTys) {
    if (!isa<RankedTensorType>(argTy))
      return emitError()
             << "semantics region arguments must all be ranked tensors."
             << "use 0-d tensors for scalars";
  }
  for (auto argTy : computeParamTys) {
    if (!argTy.isIntOrIndex())
      return emitError() << "compute parameters must be int or index "
                         << "types";
  }
  return success();
}

LogicalResult DefineOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  SmallVector<std::pair<BufferTypeInterface, unsigned>, 4> bufferArgs;
  for (auto source : getSources().getAsRange<FlatSymbolRefAttr>()) {
    auto sourceOp =
        symbolTable.lookupNearestSymbolFrom<DeclareBufferOp>(*this, source);
    if (!sourceOp)
      return emitError() << "referred source buffer '" << source
                         << "' does not exist";
    bufferArgs.push_back({sourceOp.getBufferType(), sourceOp.getSize()});
  }
  for (auto dest : getDestinations().getAsRange<FlatSymbolRefAttr>()) {
    auto destOp =
        symbolTable.lookupNearestSymbolFrom<DeclareBufferOp>(*this, dest);
    if (!destOp)
      return emitError() << "referred destination buffer '" << dest
                         << "' does not exist";
    bufferArgs.push_back({destOp.getBufferType(), destOp.getSize()});
  }
  SmallVector<BufferAccessOpInterface, 4> patterns;
  for (auto operand : getAccessBlock().getTerminator()->getOperands()) {
    auto pattern = operand.getDefiningOp<BufferAccessOpInterface>();
    assert(pattern);
    patterns.push_back(pattern);
  }
  // check compatibility
  for (auto [bufferArg, pattern] : llvm::zip(bufferArgs, patterns)) {
    auto [bufferType, bufferSize] = bufferArg;
    if (failed(pattern.verifyCompatibility(bufferType, bufferSize)))
      return emitError() << "buffer access pattern is not compatible with the "
                            "referred buffer";
  }
  return success();
}

LogicalResult EmitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto defineOp = symbolTable.lookupNearestSymbolFrom<DefineOp>(
      *this, getInstructionAttr());
  if (!defineOp)
    return emitError() << "referred instruction '" << getInstruction()
                       << "' does not exist";
  if (defineOp.getAccessBlock().getNumArguments() !=
      getStaticAddrParams().size())
    return emitError() << "number of address parameters must match the number "
                          "of access region "
                          "arguments";
  if (defineOp.getExtraComputeArgs().size() != getStaticComputeParams().size())
    return emitError() << "number of compute parameters must match the number "
                          "of semantics region "
                          "arguments";
  return success();
}