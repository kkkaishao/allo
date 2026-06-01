/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/AlloAttrs.h"

#include "allo/IR/AlloAttrs.h"
#include "mlir/CAPI/IR.h"
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
