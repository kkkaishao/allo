/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/AlloAttrs.h"

#include "allo/IR/AlloAttrs.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// PartitionAxisAttr
//===----------------------------------------------------------------------===//

bool alloAttributeIsAPartitionAxis(MlirAttribute attr) {
  return isa<allo::PartitionAxisAttr>(unwrap(attr));
}

MlirAttribute alloPartitionAxisAttrGet(MlirContext ctx, uint32_t kind,
                                       int64_t factor, int64_t dim) {
  return wrap(allo::PartitionAxisAttr::get(
      unwrap(ctx), static_cast<allo::PartitionKindEnum>(kind), factor, dim));
}

uint32_t alloPartitionAxisAttrGetKind(MlirAttribute attr) {
  return static_cast<uint32_t>(
      cast<allo::PartitionAxisAttr>(unwrap(attr)).getKind());
}

int64_t alloPartitionAxisAttrGetFactor(MlirAttribute attr) {
  return cast<allo::PartitionAxisAttr>(unwrap(attr)).getFactor();
}

int64_t alloPartitionAxisAttrGetDim(MlirAttribute attr) {
  return cast<allo::PartitionAxisAttr>(unwrap(attr)).getDim();
}

MlirTypeID alloPartitionAxisAttrGetTypeID(void) {
  return wrap(allo::PartitionAxisAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// PartitionAttr
//===----------------------------------------------------------------------===//

bool alloAttributeIsAPartition(MlirAttribute attr) {
  return isa<allo::PartitionAttr>(unwrap(attr));
}

MlirAttribute alloPartitionAttrGet(MlirContext ctx, intptr_t nAxes,
                                   MlirAttribute const *axes) {
  llvm::SmallVector<allo::PartitionAxisAttr> partitions;
  partitions.reserve(nAxes);
  for (intptr_t i = 0; i < nAxes; ++i)
    partitions.push_back(cast<allo::PartitionAxisAttr>(unwrap(axes[i])));
  return wrap(allo::PartitionAttr::get(unwrap(ctx), partitions));
}

intptr_t alloPartitionAttrGetNumAxes(MlirAttribute attr) {
  return static_cast<intptr_t>(
      cast<allo::PartitionAttr>(unwrap(attr)).getPartitions().size());
}

MlirAttribute alloPartitionAttrGetAxis(MlirAttribute attr, intptr_t pos) {
  return wrap(cast<allo::PartitionAttr>(unwrap(attr)).getPartitions()[pos]);
}

MlirTypeID alloPartitionAttrGetTypeID(void) {
  return wrap(allo::PartitionAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Enum-backed attributes: generate the four accessors isa / get(value) /
// getValue / getTypeID from (CApiName, C++ attr, C++ enum).
//===----------------------------------------------------------------------===//

#define ALLO_ENUM_ATTR_CAPI(CApiName, CppAttr, CppEnum)                        \
  bool alloAttributeIsA##CApiName(MlirAttribute attr) {                        \
    return isa<allo::CppAttr>(unwrap(attr));                                   \
  }                                                                            \
  MlirAttribute allo##CApiName##AttrGet(MlirContext ctx, uint32_t value) {     \
    return wrap(                                                               \
        allo::CppAttr::get(unwrap(ctx), static_cast<allo::CppEnum>(value)));   \
  }                                                                            \
  uint32_t allo##CApiName##AttrGetValue(MlirAttribute attr) {                  \
    return static_cast<uint32_t>(                                              \
        cast<allo::CppAttr>(unwrap(attr)).getValue());                         \
  }                                                                            \
  MlirTypeID allo##CApiName##AttrGetTypeID(void) {                             \
    return wrap(allo::CppAttr::getTypeID());                                   \
  }

ALLO_ENUM_ATTR_CAPI(AssumeDepType, AssumeDepTypeEnumAttr, AssumeDepTypeEnum)
ALLO_ENUM_ATTR_CAPI(AssumeDepDir, AssumeDepDirEnumAttr, AssumeDepDirEnum)
ALLO_ENUM_ATTR_CAPI(MemoryKind, MemoryKindEnumAttr, MemoryKindEnum)
ALLO_ENUM_ATTR_CAPI(Determinacy, DeterminacyEnumAttr, DeterminacyEnum)
ALLO_ENUM_ATTR_CAPI(CombOpKind, CombOpKindEnumAttr, CombOpKindEnum)
ALLO_ENUM_ATTR_CAPI(StallContract, StallContractEnumAttr, StallContractEnum)

#undef ALLO_ENUM_ATTR_CAPI

//===----------------------------------------------------------------------===//
// Resource cost evaluation
//===----------------------------------------------------------------------===//

void alloEvaluateResourceUse(MlirAttribute uses, intptr_t nParams,
                             const int64_t *params,
                             AlloResourceUseCallback callback, void *userData) {
  for (auto [resource, amount] : allo::evaluateResourceUse(
           dyn_cast_or_null<ArrayAttr>(unwrap(uses)),
           llvm::ArrayRef<int64_t>(params, static_cast<size_t>(nParams))))
    callback(wrap(resource.getLeafReference().getValue()), amount, userData);
}

double alloEvaluateCost(MlirAttribute cost, int64_t param) {
  auto attr = dyn_cast_or_null<allo::CostAttr>(unwrap(cost));
  return attr ? attr.evaluate(param) : 0.0;
}
