# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Scheduling)
add_public_tablegen_target(MLIRAlloSchedulingPassIncGen)

add_dependencies(MLIRAlloIncGen MLIRAlloSchedulingPassIncGen)
