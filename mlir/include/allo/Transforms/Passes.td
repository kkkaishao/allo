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

def DataflowSpawnPass : Pass<"dataflow-spawn", "ModuleOp"> {
  let summary = "Launch tagged dataflow PEs as fibers on the marl runtime.";
  let description = [{
    Rewrite the (already LLVM-lowered) sequential calls to dataflow PEs -- the
    callees tagged `allo.dataflow.pe` by lower-dataflow -- into concurrent fiber
    spawns onto the marl runtime: `allo_df_open` / `allo_df_spawn` per PE (with a
    generated `void(ptr)` thunk that reloads the PE's operands from an alloca'd
    context) / `allo_df_join` / `allo_df_close`. Run after the module is in the
    LLVM dialect (operands are pure LLVM values, casts reconciled).
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def ReuseCleanupPass : Pass<"reuse-cleanup", "ModuleOp"> {
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

def GenerateApintWrapperPass
    : Pass<"generate-apint-wrapper", "ModuleOp"> {
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
