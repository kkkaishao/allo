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

def MaterializeApintWrapperPass
    : Pass<"materialize-apint-wrapper", "ModuleOp"> {
  let summary = "Wrap an APInt-boundary top kernel with a std-width interface";
  let description = [{
    For a top kernel whose arguments/results use non-standard-width integers
    (e.g. ap_int<5>), rename it `<name>__impl` (private) and synthesize a public
    wrapper with the original name whose boundary uses standard widths (smallest
    of 8/16/32/64). The wrapper truncates inputs into ap_int temporaries, invokes
    the real kernel, and sign/zero extends results and in-out buffers back to the
    standard widths, so numpy/ctypes and the LLVM memref ABI can call it.
    Signedness is read from the `allo.signed` marker.
  }];
  let options = [
    Option<"topName", "top", "std::string", /*default=*/"\"\"",
           "Name of the top kernel to wrap.">
  ];
  let dependentDialects = [
    "mlir::allo::AlloDialect", "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect", "mlir::memref::MemRefDialect"
  ];
}

#endif // ALLO_TRANSFORMS_PASSES_TD
