#ifndef ALLO_TRANSFORMS_PASSES_TD
#define ALLO_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LocalizeGlobalStreamPass : Pass<"localize-global-stream", "ModuleOp"> {
  let summary = "Localize global stream accesses to the function level.";
  let description = "Localize global stream accesses to the function level.";
}

#endif // ALLO_TRANSFORMS_PASSES_TD
