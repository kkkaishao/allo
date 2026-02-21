#ifndef ALLO_PASSES_TD
#define ALLO_PASSES_TD

include "mlir/Pass/PassBase.td"

def ApplyVirtualMapping : Pass<"apply-virtual-mapping", "allo::KernelOp"> {
  let summary = "Apply virtual mapping to the IR";
  let description = [{
    This pass applies virtual mapping to the IR
  }];
}

#endif // ALLO_PASSES_TD