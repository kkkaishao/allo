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

using namespace mlir;
using namespace mlir::allo;

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
