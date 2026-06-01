/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C API for the Allo dialect's custom attributes, so the Python bindings can
 * build and introspect them directly.
 */

#ifndef ALLO_C_ALLOATTRS_H
#define ALLO_C_ALLOATTRS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// PartitionAxisAttr  (#allo.part_axis(dim, kind, factor))
//
// `kind` mirrors `allo::PartitionKindEnum`:
//   0 = Complete, 1 = Block, 2 = Cyclic.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsAPartitionAxis(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute alloPartitionAxisAttrGet(MlirContext ctx,
                                                          uint32_t kind,
                                                          int64_t factor,
                                                          int64_t dim);

MLIR_CAPI_EXPORTED uint32_t alloPartitionAxisAttrGetKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t alloPartitionAxisAttrGetFactor(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t alloPartitionAxisAttrGetDim(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// PartitionAttr  (#allo.partition<[ axes... ]>)
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsAPartition(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute alloPartitionAttrGet(
    MlirContext ctx, intptr_t nAxes, MlirAttribute const *axes);

MLIR_CAPI_EXPORTED intptr_t alloPartitionAttrGetNumAxes(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloPartitionAttrGetAxis(MlirAttribute attr,
                                                          intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_ALLOATTRS_H
