/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/IR/IntegerSet.h"

#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloOps.h"
#include "allo/IR/AlloTypes.h"

#include "allo/IR/AlloDialect.cpp.inc"

#include "allo/IR/AlloEnums.cpp.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "allo/IR/AlloAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "allo/IR/AlloTypes.cpp.inc"

using namespace mlir;
using namespace mlir::allo;

static LogicalResult
verifyNoDuplicateSymbols(ArrayAttr symbols, StringRef symbolKind,
                         llvm::function_ref<InFlightDiagnostic()> emitError) {
  llvm::SmallDenseSet<StringRef> seenSymbols;
  for (auto sym : symbols) {
    auto name = cast<StringAttr>(sym).getValue();
    if (!seenSymbols.insert(name).second)
      return emitError() << "duplicate " << symbolKind << " symbol '" << sym
                         << "'";
  }
  return success();
}

static LogicalResult
appendShapedTypes(ArrayAttr attrs, StringRef typeKind,
                  SmallVectorImpl<ShapedType> &sigTypes,
                  llvm::function_ref<InFlightDiagnostic()> emitError) {
  for (auto attr : attrs.getAsRange<TypeAttr>()) {
    Type ty = attr.getValue();
    if (!isa<MemRefType, RankedTensorType>(ty))
      return emitError() << typeKind
                         << " type must be memref or tensor, but got " << ty;
    sigTypes.push_back(cast<ShapedType>(ty));
  }
  return success();
}

static LogicalResult verifySemanticsArgAgainstSig(
    Type argTy, ShapedType sig,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  auto shapedArg = dyn_cast<ShapedType>(argTy);
  if (!shapedArg)
    return emitError() << "semantics block arguments must be of shaped type";
  if (shapedArg.getRank() != sig.getRank()) {
    return emitError() << "semantics block argument has rank "
                       << shapedArg.getRank()
                       << ", but corresponding operand/result has rank "
                       << sig.getRank();
  }
  if (shapedArg.getElementType() != sig.getElementType()) {
    return emitError() << "semantics block argument has element type "
                       << shapedArg.getElementType()
                       << ", but corresponding operand/result has element type "
                       << sig.getElementType();
  }
  return success();
}

static LogicalResult verifyYieldTypeAgainstResult(
    Type yieldTy, Type resultTy,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  auto shapedYield = dyn_cast<ShapedType>(yieldTy);
  auto shapedResult = cast<ShapedType>(resultTy);
  if (!shapedYield)
    return emitError() << "yield operands must be of shaped type";
  if (shapedYield.getRank() != shapedResult.getRank()) {
    return emitError() << "yield operand has rank " << shapedYield.getRank()
                       << ", but corresponding result has rank "
                       << shapedResult.getRank();
  }
  if (shapedYield.getElementType() != shapedResult.getElementType()) {
    return emitError() << "yield operand has element type "
                       << shapedYield.getElementType()
                       << ", but corresponding result has element type "
                       << shapedResult.getElementType();
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
  auto streamTy = getStream().getType();
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
  auto streamTy = getStream().getType();
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
  auto streamTy = globalStream.getStream().getType();
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

LogicalResult
GlobalStreamPutOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto globalStream = symbolTable.lookupNearestSymbolFrom<GlobalStreamCreateOp>(
      *this, getStreamAttr());
  if (!globalStream) {
    return emitOpError() << "referenced global stream '" << getStream()
                         << "' does not exist";
  }
  auto streamTy = globalStream.getStream().getType();
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

LogicalResult InstructionDefineOp::verify() {
  // Cond 1: no duplicate symbol names
  if (failed(verifyNoDuplicateSymbols(getIndexSymbols(), "index",
                                      [this] { return emitError(); })))
    return failure();
  if (failed(verifyNoDuplicateSymbols(getDomainSymbols(), "domain",
                                      [this] { return emitError(); })))
    return failure();

  // Cond 2: input/output types must be memref/tensor
  SmallVector<ShapedType, 4> sigTypes;
  if (failed(appendShapedTypes(getSourceTypes(), "source", sigTypes,
                               [this] { return emitError(); })))
    return failure();
  if (failed(appendShapedTypes(getDestinationTypes(), "target", sigTypes,
                               [this] { return emitError(); })))
    return failure();
  // Cond 3: dim count of indexing maps must match operand/result rank
  unsigned nMapSyms = getIndexSymbols().size();
  if (getIndexMaps().size() != sigTypes.size()) {
    return emitError()
           << "number of indexing maps (" << getIndexMaps().size()
           << ") does not match total number of operands and results ("
           << sigTypes.size() << ")";
  }
  for (auto [mapAttr, sig] : llvm::zip(getIndexMaps(), sigTypes)) {
    auto map = cast<AffineMapAttr>(mapAttr).getValue();
    if (map.getNumDims() != sig.getRank()) {
      return emitError()
             << "indexing map has " << map.getNumDims()
             << " dimensions, but corresponding operand/result has rank "
             << sig.getRank();
    }
    if (map.getNumSymbols() != nMapSyms) {
      return emitError() << "indexing map has " << map.getNumSymbols()
                         << " symbols, but instruction defines " << nMapSyms
                         << " index symbols";
    }
  }
  // Cond 4: dim count of indexing sets must match operand/result rank
  unsigned nSetSyms = getDomainSymbols().size();
  if (getIndexSets().size() != sigTypes.size()) {
    return emitError()
           << "number of indexing sets (" << getIndexSets().size()
           << ") does not match total number of operands and results ("
           << sigTypes.size() << ")";
  }
  for (auto [setAttr, sig] : llvm::zip(getIndexSets(), sigTypes)) {
    auto set = cast<IntegerSetAttr>(setAttr).getValue();
    if (set.getNumDims() != sig.getRank()) {
      return emitError()
             << "indexing set has " << set.getNumDims()
             << " elements, but corresponding operand/result has rank "
             << sig.getRank();
    }
    if (set.getNumSymbols() != nSetSyms) {
      return emitError() << "indexing set has " << set.getNumSymbols()
                         << " symbols, but instruction defines " << nSetSyms
                         << " domain symbols";
    }
  }
  // Cond 5: semantics block must end with InstructionYieldOp
  Block &sem = getRegion().front();
  if (sem.empty() || !isa<InstructionYieldOp>(sem.back())) {
    return emitError() << "semantics block must end with InstructionYieldOp";
  }
  // Cond 6: semantics block args must match operand types
  auto semArgs = sem.getArguments();
  if (semArgs.size() != sigTypes.size()) {
    return emitError() << "number of semantics block arguments ("
                       << semArgs.size()
                       << ") does not match number of operands in signature ("
                       << sigTypes.size() << ")";
  }
  for (auto [arg, sig] : llvm::zip(semArgs, sigTypes)) {
    // We only require rank and element type to match because semantics describe
    // computation relation among operands/results.
    if (failed(verifySemanticsArgAgainstSig(arg.getType(), sig,
                                            [this] { return emitError(); })))
      return failure();
  }
  // Cond 7: yield types must match instruction results
  auto yieldOp = cast<InstructionYieldOp>(sem.back());
  auto yieldTypes = yieldOp.getOperandTypes();
  auto resultTypes = getDestinationTypes();
  if (yieldTypes.size() != resultTypes.size()) {
    return emitError() << "number of yield operands (" << yieldTypes.size()
                       << ") does not match number of results in signature ("
                       << resultTypes.size() << ")";
  }
  for (auto [yieldTy, resultTy] : llvm::zip(yieldTypes, resultTypes)) {
    if (failed(verifyYieldTypeAgainstResult(yieldTy,
                                            cast<TypeAttr>(resultTy).getValue(),
                                            [this] { return emitError(); })))
      return failure();
  }
  return success();
}

LogicalResult
InstructionEmitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Cond 1: referenced instruction must exist
  auto defineOp = symbolTable.lookupNearestSymbolFrom<InstructionDefineOp>(
      *this, getInstructionAttr());
  if (!defineOp)
    return emitError() << "referenced instruction '" << getInstruction()
                       << "' does not exist";

  // Cond 2: source types must match instruction signature
  auto definedTypes =
      llvm::to_vector<4>(defineOp.getSourceTypes().getAsRange<TypeAttr>());
  llvm::append_range(definedTypes,
                     defineOp.getDestinationTypes().getAsRange<TypeAttr>());
  auto actualTypes = getOperandTypes();
  if (actualTypes.size() != definedTypes.size()) {
    return emitError()
           << "number of operands (" << actualTypes.size()
           << ") does not match total number of operands and results in "
              "instruction signature ("
           << definedTypes.size() << ")";
  }
  for (auto [actual, defined] : llvm::zip(actualTypes, definedTypes)) {
    if (actual != cast<TypeAttr>(defined).getValue()) {
      return emitError() << "operand type " << actual
                         << " does not match expected type " << defined
                         << " in instruction signature";
    }
  }

  // Cond 3: every symbol value must have a corresponding symbol in the
  // instruction definition
  llvm::SmallDenseSet<StringAttr> syms;
  for (auto sym : defineOp.getIndexSymbols())
    syms.insert(cast<StringAttr>(sym));
  for (auto sym : defineOp.getDomainSymbols())
    syms.insert(cast<StringAttr>(sym));
  for (auto sym : getSymbolValues()) {
    auto name = sym.getName();
    if (!syms.contains(name)) {
      return emitError() << "symbol '" << name
                         << "' is not defined in the "
                            "referenced instruction definition";
    }
    if (!isa<IntegerAttr>(sym.getValue()))
      return emitError() << "symbol value must be an integer attribute";
  }
  return success();
}

LogicalResult InstructionEmitOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  auto resTypes = adaptor.getDestinations().getTypes();
  // memref is not of value semantics
  auto tensorRange = llvm::make_filter_range(
      resTypes, [](Type ty) { return isa<RankedTensorType>(ty); });
  llvm::append_range(inferredReturnTypes, tensorRange);
  return success();
}
