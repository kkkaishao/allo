#ifndef ALLO_SCHEDULING_PASSES_TD
#define ALLO_SCHEDULING_PASSES_TD

include "mlir/Pass/PassBase.td"

def AlloSchedulePass : Pass<"allo-schedule", "func::FuncOp"> {
  let summary = "SDC-schedule affine loops, carrying the result as attributes";
  let description = [{
    Analyze memory and stream dependences in a `func.func`, solve an SDC
    scheduling problem for each innermost affine loop, and annotate the solution
    (per-op start times, per-region initiation interval) onto the IR as
    attributes. Unlike `convert-loop-to-schedule`, this pass materializes no
    structural IR: the schedule is a decision carried on the ops for a later
    export/materialization stage.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"dumpRegionGraph", "dump-region-graph", "bool", "false",
           "Print the coarse cross-region dependence graph as DOT to stderr">
  ];
}

#endif // ALLO_SCHEDULING_PASSES_TD
