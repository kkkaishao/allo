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

//===----------------------------------------------------------------------===//
// AssumeDepTypeAttr  (#allo<dep_type inter|intra>)
//
// `value` mirrors `allo::AssumeDepTypeEnum`: 0 = Inter, 1 = Intra.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsAAssumeDepType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloAssumeDepTypeAttrGet(MlirContext ctx,
                                                          uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloAssumeDepTypeAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// AssumeDepDirAttr  (#allo<dep_dir raw|war|waw>)
//
// `value` mirrors `allo::AssumeDepDirEnum`: 0 = RAW, 1 = WAR, 2 = WAW.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsAAssumeDepDir(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloAssumeDepDirAttrGet(MlirContext ctx,
                                                         uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloAssumeDepDirAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_ALLOATTRS_H
