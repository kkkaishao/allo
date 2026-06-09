#ifndef ALLO_CONVERSION_PASSES_TD
#define ALLO_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertAlloToFuncPass : Pass<"convert-allo-to-func"> {
  let summary = "Convert allo dialect to func dialect";
  let description = [{Convert allo dialect to func dialect}];
}

def LowerInstructionsPass
    : Pass<"lower-instructions", "ModuleOp"> {
  let summary = "Inline allo instruction semantics into linalg/tensor/arith";
  let description = [{
    The oracle lowering. Lowers allo.buffer to memref.global and inlines each
    allo.emit's instruction semantics in place (reading operand slices, running
    the compute region, storing the result slice back to the buffer global), so
    the program can be lowered to LLVM and executed. Each emit is self-contained
    over mutable memref globals, so emits nested in control flow are supported.
  }];
}

def LowerDataflowPass : Pass<"lower-dataflow", "ModuleOp"> {
  let summary = "Lower local scalar streams to the host dataflow simulator ABI";
  let description = [{
    Lower Allo local scalar stream operations to calls into the host dataflow
    simulator runtime and tag stream-connected kernels as dataflow PEs, which the
    late `allo-dataflow-spawn` pass launches as concurrent fibers on the marl
    runtime.
  }];
}

#endif // ALLO_CONVERSION_PASSES_TD
