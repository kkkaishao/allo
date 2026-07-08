# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Public C API for the Allo dialect, passes, schedule analysis and transform
# interpreter. This is the only surface the upstream-bindings Python extension
# (`allo._mlir._mlir_libs._allo`) links against, so every Allo / MLIR-core C++
# entry point the frontend needs is funnelled through here.
add_mlir_public_c_api_library(MLIRAlloCAPI
    Registration.cpp
    Passes.cpp
    Schedule.cpp
    IRUtils.cpp
    AlloTypes.cpp
    AlloAttrs.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRCAPIIR
    MLIRSupport
    MLIRAllo
    MLIRAlloConversion
    MLIRAlloMicroarch
    MLIRAlloScheduling
    MLIRAlloTransforms
    MLIRAlloTransformOps
    MLIRAlloTranslation
)
