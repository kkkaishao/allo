#ifndef ACT_CONVERSION_PASSES_TD
#define ACT_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertActToCanonicalFormPass : Pass<"convert-act-to-canonical-form"> {
  let summary = "Convert ACT dialect to canonical form";
  let description = [{}];
}

def ConvertCanonicalFormToActPass : Pass<"convert-canonical-form-to-act"> {
  let summary = "Convert canonical form to ACT dialect";
  let description = [{}];
  let options = [
    Option<"isaPath", "isa-path", "std::optional<std::string>", "std::nullopt",
           "Path to the ISA definition file.">,
  ];
}

#endif // ACT_CONVERSION_PASSES_TD