#ifndef ALLO_CONVERSION_PASSES_TD
#define ALLO_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertAlloToFuncPass : Pass<"convert-allo-to-func"> {
  let summary = "Convert allo dialect to func dialect";
  let description = [{Convert allo dialect to func dialect}];
}

#endif // ALLO_CONVERSION_PASSES_TD
