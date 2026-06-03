#ifndef ALLO_TRANSFORMS_PASSES_TD
#define ALLO_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def MaterializeTopologyPass : Pass<"materialize-topology", "ModuleOp"> {
  let summary = "Materialize static stream topology.";
  let description = [{
    Specialize mapped process interfaces by scalarizing statically indexed
    ranked stream lanes and trimming unused process arguments.
  }];
  let options = [
    Option<"allowPublicAbiChange", "allow-public-abi-change", "bool", "false",
           "Allow scalarizing ranked stream ports on public kernels.">
  ];
}

def GridMappingPass : Pass<"grid-mapping", "ModuleOp"> {
  let summary = "Map processes to a grid.";
  let description = [{
    Map processes to a grid.
  }];
}

def ReuseCleanupPass : Pass<"allo-reuse-cleanup", "ModuleOp"> {
  let summary = "Simplify the IR produced by reuse_at.";
  let description = [{
    Clean up the redundant IR left behind by `transform.allo.reuse_at`: run
    CSE + canonicalization, merge adjacent `affine.if` wrappers that share a
    condition (the per-access conditional loads), and sink pure tails into the
    branches. This is a no-op on modules that contain no reuse_at output, which
    is detected via the `allo.reuse.maintenance` markers; those internal markers
    are stripped once cleanup completes.
  }];
  let dependentDialects = ["affine::AffineDialect"];
}

#endif // ALLO_TRANSFORMS_PASSES_TD
