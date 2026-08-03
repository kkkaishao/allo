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

def LowerHLMemPass : Pass<"allo-lower-hlmem", "circt::hw::HWModuleOp"> {
  let summary = "Lower seq.hlmem and its ports onto an sv.reg array";
  let description = [{
    Replaces CIRCT's `lower-seq-hlmem`, which materializes every write of a
    memory inside one `always_ff` and so cannot describe a port per `always`
    block, the only shape a true-dual-port block RAM infers from. Also gives the
    array its power-on contents, which the memory carries and the lowering would
    otherwise erase.
  }];
  let dependentDialects = [
    "circt::hw::HWDialect", "circt::sv::SVDialect", "circt::seq::SeqDialect"
  ];
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
