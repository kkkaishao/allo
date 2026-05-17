#ifndef ALLO_CONVERSION_PASSES_TD
#define ALLO_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertAlloToFuncPass : Pass<"convert-allo-to-func"> {
  let summary = "Convert allo dialect to func dialect";
  let description = [{Convert allo dialect to func dialect}];
}

def LowerDataflowPass : Pass<"lower-dataflow", "ModuleOp"> {
  let summary = "Lower local scalar streams to the host dataflow simulator ABI";
  let description = [{
    Lower Allo local scalar stream operations to calls into the host dataflow
    simulator runtime and wrap stream-connected kernel invocations with OpenMP
    sections.
  }];
  let options = [
    Option<"numThreads", "num-threads", "unsigned", "8",
           "The number of threads to use for dataflow simulation.">
  ];
}

#endif // ALLO_CONVERSION_PASSES_TD
