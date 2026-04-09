#ifndef ACT_CONVERSION_PASSES_TD
#define ACT_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertActToCanonicalFormPass : Pass<"convert-act-to-canonical-form"> {
  let summary = "Convert ACT dialect to canonical form";
  let description = [{}];
  let options = [
    Option<"enableTensor", "enable-tensor", "bool", "true",
           "Enable conversion to tensor types instead of memref types.">,
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
  ];
}

#endif // ACT_CONVERSION_PASSES_TD