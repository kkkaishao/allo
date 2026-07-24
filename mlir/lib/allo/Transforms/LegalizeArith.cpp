/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::allo {
#define GEN_PASS_DEF_LEGALIZEARITHPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {

// The RTL-path, device-IP-aware replacement for `arith-expand`. A composite
// arith op the device can realize directly (a matching `dcp.operator`) is KEPT,
// so the scheduler binds it to that IP; every other one is EXPANDED into
// primitive arith by the upstream arith-expand patterns. Integer max/min are
// native combinational ops and are left alone (never marked illegal).
struct LegalizeArithPass
    : public allo::impl::LegalizeArithPassBase<LegalizeArithPass> {
  using LegalizeArithPassBase::LegalizeArithPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Built from the injected `dcp.device` / `dcp.operator` IR
    OperatorLibrary lib = OperatorLibrary::fromModule(module);

    // Reuse the upstream expansion patterns
    RewritePatternSet patterns(&getContext());
    arith::populateArithExpandOpsPatterns(patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();

    // A composite op is legal (kept) iff the device realizes it directly;
    // otherwise it is illegal and the patterns decompose it into primitives.
    auto keepIfRealizable = [&lib](Operation *op) {
      return lib.hasDirectRealization(op);
    };
    target.addDynamicallyLegalOp<arith::CeilDivSIOp, arith::CeilDivUIOp,
                                 arith::FloorDivSIOp, arith::MaximumFOp,
                                 arith::MinimumFOp, arith::MaxNumFOp,
                                 arith::MinNumFOp>(keepIfRealizable);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
