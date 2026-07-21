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
MLIR_CAPI_EXPORTED MlirTypeID alloPartitionAxisAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// PartitionAttr  (#allo.partition<[ axes... ]>)
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsAPartition(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute alloPartitionAttrGet(
    MlirContext ctx, intptr_t nAxes, MlirAttribute const *axes);

MLIR_CAPI_EXPORTED intptr_t alloPartitionAttrGetNumAxes(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloPartitionAttrGetAxis(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED MlirTypeID alloPartitionAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Enum-backed attributes. `value` is the underlying I32 enum case (see the
// per-attr comments below and AlloAttrs.td), and Get()/GetValue() round-trip
// it. All four accessors follow the same shape for every enum attr.
//===----------------------------------------------------------------------===//

// AssumeDepTypeAttr (#allo<dep_type inter|intra>): 0 = Inter, 1 = Intra.
MLIR_CAPI_EXPORTED bool alloAttributeIsAAssumeDepType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloAssumeDepTypeAttrGet(MlirContext ctx,
                                                          uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloAssumeDepTypeAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloAssumeDepTypeAttrGetTypeID(void);

// AssumeDepDirAttr (#allo<dep_dir raw|war|waw>): 0 = RAW, 1 = WAR, 2 = WAW.
MLIR_CAPI_EXPORTED bool alloAttributeIsAAssumeDepDir(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloAssumeDepDirAttrGet(MlirContext ctx,
                                                         uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloAssumeDepDirAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloAssumeDepDirAttrGetTypeID(void);

// MemoryImplAttr (#allo<mem_impl ...>): 0=Auto 1=Register 2=LUTRAM 3=BRAM
// 4=URAM.
MLIR_CAPI_EXPORTED bool alloAttributeIsAMemoryImpl(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloMemoryImplAttrGet(MlirContext ctx,
                                                       uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloMemoryImplAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloMemoryImplAttrGetTypeID(void);

// MemoryPortAttr (#allo<mem_port ...>): 0=SinglePort 1=SimpleDualPort
// 2=TrueDualPort.
MLIR_CAPI_EXPORTED bool alloAttributeIsAMemoryPort(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloMemoryPortAttrGet(MlirContext ctx,
                                                       uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloMemoryPortAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloMemoryPortAttrGetTypeID(void);

// MemoryKindAttr (#allo<mem_kind ram|rom>): 0 = RAM, 1 = ROM.
MLIR_CAPI_EXPORTED bool alloAttributeIsAMemoryKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloMemoryKindAttrGet(MlirContext ctx,
                                                       uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloMemoryKindAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloMemoryKindAttrGetTypeID(void);

// DeterminacyAttr (#allo<determinacy ...>):
//   0=CountedStatic 1=Conditional 2=Indeterminate 3=Concurrent.
MLIR_CAPI_EXPORTED bool alloAttributeIsADeterminacy(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloDeterminacyAttrGet(MlirContext ctx,
                                                        uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloDeterminacyAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloDeterminacyAttrGetTypeID(void);

// CombOpKindAttr (#allo<comb_kind ...>): the CombOpKindEnum case (see the .td).
MLIR_CAPI_EXPORTED bool alloAttributeIsACombOpKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloCombOpKindAttrGet(MlirContext ctx,
                                                       uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloCombOpKindAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloCombOpKindAttrGetTypeID(void);

// StallContractAttr (#allo<stall ce|free|elastic>): 0 = Ce, 1 = Free, 2 =
// Elastic.
MLIR_CAPI_EXPORTED bool alloAttributeIsAStallContract(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloStallContractAttrGet(MlirContext ctx,
                                                          uint32_t value);
MLIR_CAPI_EXPORTED uint32_t alloStallContractAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloStallContractAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_ALLOATTRS_H
