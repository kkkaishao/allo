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

#endif // ALLO_TRANSFORMS_PASSES_TD
