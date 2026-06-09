# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(MLIRAlloTransformsIncGen)

add_dependencies(MLIRAlloIncGen MLIRAlloTransformsIncGen)
