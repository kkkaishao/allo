# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloTranslation
  VivadoHLSEmitter.cpp

  DEPENDS
  MLIRAlloIncGen

  LINK_LIBS PUBLIC
  MLIRAllo
)
