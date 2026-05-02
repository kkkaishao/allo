/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_PYTHON_IR_H
#define ALLO_PYTHON_IR_H

#include "mlir/IR/Builders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

class AlloOpBuilder : public mlir::OpBuilder {
public:
  using OpBuilder::OpBuilder;
  mlir::Location getLocation() const { return loc; }
  void setLocation(mlir::Location newLoc) { loc = newLoc; }
  void setUnknownLoc() { loc = getUnknownLoc(); }
  std::pair<OpBuilder::InsertPoint, mlir::Location>
  getInsertionPointAndLoc() const {
    return {saveInsertionPoint(), loc};
  }
  void setInsertionPointAndLoc(const OpBuilder::InsertPoint &ip,
                               mlir::Location newLoc) {
    restoreInsertionPoint(ip);
    loc = newLoc;
  }

private:
  // default init to unknown
  mlir::Location loc = getUnknownLoc();
};

void bindIR(nb::module_ &m);
void bindMathOps(nb::module_ &m);
void bindArithOps(nb::module_ &m);
void bindSCFOps(nb::module_ &m);
void bindCFOps(nb::module_ &m);
void bindFuncOps(nb::module_ &m);
void bindAffineOps(nb::module_ &m);
void bindTensorOps(nb::module_ &m);
void bindMemRefOps(nb::module_ &m);
void bindLinalgOps(nb::module_ &m);
void bindTransform(nb::module_ &m);
void bindUtils(nb::module_ &m);
void bindUBOps(nb::module_ &m);
void bindAlloOps(nb::module_ &m);
void bindPasses(nb::module_ &m);

inline mlir::OpPrintingFlags getOpPrintingFlags(bool debug = false) {
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.enableDebugInfo(debug);
  printingFlags.printNameLocAsPrefix(true);
  printingFlags.printGenericOpForm(false);
  return printingFlags;
}

#endif // ALLO_PYTHON_IR_H
