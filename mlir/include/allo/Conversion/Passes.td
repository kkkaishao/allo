/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
    simulator runtime and tag stream-connected kernels as dataflow PEs, which
    the late `allo-dataflow-spawn` pass launches as concurrent fibers on the
    marl runtime.
  }];
}

#endif // ALLO_CONVERSION_PASSES_TD
