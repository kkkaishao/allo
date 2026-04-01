#ifndef ALLO_CONVERSION_PASSES_TD
#define ALLO_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertInstructionToCanonicalFormPass : Pass<"convert-instruction-to-canonical-form", "ModuleOp"> {
  let summary = "Convert instruction emission to canonical form";
  let description = [{
    This pass converts the instruction emission operation (`instr.emit`) into a more canonical form,
    which is composed of a set of primitive operations that are used to express the same semantics.
  }];
  let options = [
    Option<"stripDefinitions", "strip-definitions", "bool", /*default=*/"false",
        "Whether to strip definitions from the emitted instructions">
  ];
}

#endif // ALLO_CONVERSION_PASSES_TD