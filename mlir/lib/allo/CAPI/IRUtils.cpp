/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/IRUtils.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/Block.h"

using namespace mlir;

void alloBlockErase(MlirBlock block) { unwrap(block)->erase(); }

void alloBlockMergeBefore(MlirBlock src, MlirBlock dst) {
  Block *s = unwrap(src);
  Block *d = unwrap(dst);
  auto insertPt = d->empty() ? d->end() : std::prev(d->end());
  d->getOperations().splice(insertPt, s->getOperations());
  s->erase();
}

MlirModule alloCloneModuleOp(MlirModule module) {
  return wrap(unwrap(module).clone());
}
