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

def FoldConstantCallsPass : Pass<"fold-constant-calls", "ModuleOp"> {
  let summary = "Evaluate lazy-consteval kernel invokes with constant arguments.";
  let description = [{
    Lowers the "lazy consteval" feature. A `@consteval(lazy=True)` function
    enters the IR as an `allo.kernel` tagged `allo.lazy` (fast filter) and is
    called via `allo.invoke`. Once grid-mapping has specialized the call sites so
    every invoke argument is constant, this pass evaluates each invoke at compile
    time, replaces it with the constant result, and deletes the now-unused lazy
    kernels -- so they never reach codegen. Evaluation clones the kernel body
    into a throwaway func.func, substitutes the constant arguments, and folds it
    to a constant via a fixpoint of canonicalize + CSE + affine store-to-load
    forwarding (so local scratch arrays fold) + full unrolling of constant-trip
    affine.for and scf.for loops. Because `@consteval(lazy=True)` is an explicit
    request to fold the function away, an invoke that cannot be reduced to a
    constant (non-constant arguments or a body that does not evaluate) is a hard
    error. Run between grid-mapping and materialize-topology.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::affine::AffineDialect"
  ];
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

def FoldIfStatementsPass
    : Pass<"fold-if-statements", "func::FuncOp"> {
  let summary = "Fold away affine.if / scf.if the scheduler cannot pipeline across";
  let description = [{
    A collection of rewrites that remove or simplify `if` control flow so the SDC
    scheduler sees as little of it as possible inside a loop body. Greedily
    applied; the home for all `if`-oriented transforms. Currently:

    * **Hyperblock predication.** Flatten an `affine.if`/`scf.if` into a single
      predicated (hyperblock) datapath. The branch condition is an `i1`
      (materialized from the `affine.if` integer set, or taken directly from
      `scf.if`); both branches are speculated -- side-effect-free ops and loads
      hoisted unconditionally, value results become `arith.select`, each guarded
      store a predicated read-modify-write (`store select(cond, v, load)`). An
      `if` whose body holds an un-maskable side effect (stream get/put), a nested
      region (loop / inner if), or any other effecting op is not predicated.
    * **Guard-to-bound (affine index-set splitting).** When an `affine.if` guards
      the *entire* body of its enclosing `affine.for` with a single inequality
      that is affine in the loop's induction variable, fold the guard into the
      loop's bounds (`for j in [0,N): if j > i` -> `for j in [i+1,N)`) and drop
      the `if`. This eliminates the control construct entirely -- the dead
      iterations are skipped, not just masked -- and only applies to `affine.if`
      (its predicate is a static function of the IVs; a data-dependent `scf.if`
      cannot be folded this way).

    An `if` left standing after these rewrites schedules as one opaque unit (the
    scheduler keeps it as a raw guard around its decomposed body).
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"
  ];
}

def RaiseCountedWhilePass
    : Pass<"raise-counted-while", "func::FuncOp"> {
  let summary = "Raise a counted scf.while to scf.for";
  let description = [{
    Raise an `scf.while` whose exit is a pure ordered test of a monotonically
    increasing induction variable against a loop-invariant bound to an
    `scf.for`, so the counted-loop scheduler handles it unchanged. Only
    provably-counted, increasing loops are raised (`<`/`<=`, constant positive
    step, `index` IV, unused IV result, identity condition forwarding, a
    test-only `before` region); every other `scf.while` is left untouched for
    the uncounted (flushable-pipeline) path.
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect", "mlir::arith::ArithDialect"
  ];
}

def FlattenPerfectLoopsPass
    : Pass<"flatten-perfect-loops", "func::FuncOp"> {
  let summary = "Coalesce constant-trip perfect affine loop nests into one loop";
  let description = [{
    Coalesce the maximal normalized (lower bound 0, step 1), constant-trip-count
    prefix of each perfect affine loop band into a single loop (via
    `affine::coalesceLoops`), so the SDC scheduler pipelines the whole nest at
    one II instead of scheduling only the innermost body and paying the pipeline
    fill/drain at every inner-loop re-entry. A loop with a symbolic bound is left
    alone: the coalesced induction-variable arithmetic (`floordiv`/`mod` by the
    bound) would be semi-affine and defeat the dependence analysis. A loop
    carrying `iter_args` is also left alone: coalescing merges the induction
    spaces but drops loop-carried values, which would silently miscompile an
    inner accumulator. The pass only handles simple nests, and does not attempt
    to convert a non-perfect nest into a perfect one (e.g. by peeling or
    sinking); complex nests should be flattened by the `transform.allo.flatten`
    op.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect", "mlir::scf::SCFDialect"
  ];
}

def FlattenDcpMemrefPass
    : Pass<"dcp-flatten-memref", "func::FuncOp"> {
  let summary = "Linearize scheduled DCP memory addresses into a flat 1-D index";
  let description = [{
    A DCP-normalization pass (runs after `convert-schedule-to-dcp`, before
    `allo-datapath-to-hw`). `flatten-perfect-loops` coalesces a perfect nest into
    one induction variable and delinearizes each subscript against it
    (`A[iv floordiv N, iv mod N]`), so the reified `dcp.load`/`dcp.store` carries a
    `floordiv`/`mod` address map the datapath emitter cannot lower. Since the
    backend's memories are already flat, row-major storage, this pass composes the
    memref's row-major linearization into each access map, folding the
    `floordiv`/`mod` back into the plain linear index (`A_flat[iv]`) -- the
    delinearize/linearize pair cancels, so the emitted address is the bare
    counter, matching Vitis's flat-array addressing. Only maps that actually carry
    a `floordiv`/`mod` are touched; a plain multi-dim map is already linearized by
    the emitter. A residual `floordiv`/`mod` after simplification (a transposed or
    non-row-major access) is left for the banking pass and its bit-slice / crossbar
    lowering. The memref type itself is left as the flat backing store (a
    downstream type-rewrite finalizer is deferred).
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

def ResolveBankingPass
    : Pass<"dcp-resolve-banking", "func::FuncOp"> {
  let summary = "Split a statically-banked partitioned array into per-bank memrefs";
  let description = [{
    A DCP-normalization pass (runs after `convert-schedule-to-dcp`, before
    `dcp-flatten-memref`). An `allo.part` cyclic partition asks for N physical
    banks; this pass materializes them for an **internal** array (a
    `memref.alloc`/`alloca`) whose every access is *statically banked* -- the
    partition-dim subscript is iteration-invariant modulo the factor, so each
    access hits a compile-time-constant bank (`MemoryModel::staticBank`, the same
    predicate the scheduler bound its per-bank ResII against). It creates one
    per-bank alloc (bank-shaped, no partition attribute) and routes each access to
    its bank with the in-bank offset (`i_d floordiv F`), reusing the scheduler's
    static-bank decision so the physical banks match. The emitter then sees N
    plain (`numBanks == 1`) memories -- no banking logic. A partitioned array with
    a data-dependent bank (or an external/argument array) is left untouched, so it
    reaches the emitter still `numBanks > 1` -- the crossbar fallback.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::memref::MemRefDialect"
  ];
}

def UnrollUnderPipelinePass
    : Pass<"unroll-under-pipeline", "func::FuncOp"> {
  let summary = "Fully unroll the loops nested inside a pipelined loop";
  let description = [{
    Pipelining a loop implies its body is a single straight-line datapath
    clocked at one II, so every loop nested inside a pipelined loop
    (`allo.pipeline.ii` != -1) must be fully unrolled into it -- matching Vitis,
    where `#pragma HLS pipeline` on an outer loop auto-unrolls the inner loops.
    The now-single pipelined loop is then scheduled by the existing
    innermost-pipeline path at its target II.

    A nested loop that is not a constant-trip counted loop (a dynamic bound, or
    an `scf.while`) cannot be unrolled -- the pipeline cannot be formed -- so the
    loop is left unchanged with a remark (falls back to scheduling the innermost).
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::scf::SCFDialect"
  ];
}

def PerfectizeLoopNestPass
    : Pass<"perfectize-loop-nest", "func::FuncOp"> {
  let summary = "Perfectize an imperfect loop nest by sinking prologue/epilogue";
  let description = [{
    Turn an imperfect loop nest -- an outer counted loop (`affine.for`/`scf.for`)
    whose body is `{ prologue ops..., inner loop, epilogue ops... }` -- into a
    perfect one so the existing perfect-nest scheduler pipelines it (otherwise the
    driver skips the whole nest, losing even the inner loop). The surrounding ops
    are sunk into the inner loop: the epilogue under a last-iteration guard
    (remapping the inner loop's results to its yielded values); a prologue store
    under a first-iteration guard (runs once, e.g. an accumulator init); a pure
    prologue op / a load of a memref not written in the nest unguarded
    (recompute). The guard is an `affine.if` (constant IV) for an affine.for inner
    or an `scf.if` (runtime `cmpi`) for an scf.for inner;
    `fold-if-statements` predicates it.

    Bails (leaves the nest untouched) on: sibling inner loops, an scf.while, an
    escaping/aliasing value, or other side effects.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

def ReassociateReductionsPass
    : Pass<"reassociate-reductions", "func::FuncOp"> {
  let summary = "Rebalance associative reduction chains into logarithmic trees";
  let description = [{
    Rebalance a maximal single-use chain of one associative operator into a
    balanced binary tree, so the SDC scheduler sees logarithmic depth instead of
    a linear dependence chain. Physical loop unrolling turns an accumulation
    `acc += x[i]` into a left-leaning chain `acc = ((((acc + x0) + x1) + x2) +
    x3)`, whose loop-carried accumulator threads through every operator; the
    recurrence, and hence the modulo initiation interval, then grows with the
    unroll factor. This pass folds a loop-carried accumulator in last so its
    recurrence spans a single operator, and balances a straight-line (fully
    unrolled) chain to cut its datapath depth from linear to logarithmic.

    Floating-point reassociation is not bit-exact, so the pass is gated behind
    the `float-reassoc` option (mirroring Vitis's fp-relaxed) and is a no-op
    without it. Integer reductions are exactly reassociable but bit-growth wraps
    their accumulator in `extsi`/`trunc`, so this pass leaves them unchanged.
  }];
  let options = [
    Option<"floatReassoc", "float-reassoc", "bool", "false",
           "Permit reassociation of floating-point reductions (inexact).">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect"
  ];
}

def RotateReductionsPass
    : Pass<"rotate-reductions", "func::FuncOp"> {
  let summary = "Rotate a reduction across N accumulators to reach II=1";
  let description = [{
    Replace a loop-carried float reduction's single accumulator with N
    accumulators in a shift register, so a given partial sum is updated once
    every N iterations. This turns the distance-1 recurrence (which bounds the
    modulo II at the operator latency L) into a distance-N recurrence over one
    pipelined operator, i.e. `RecII = ceil(L / N)` -- which is 1 once `N >= L`.
    The N partial sums are combined by a balanced tree after the loop.

    Only float `add`/`mul` reductions are rotated (reassociation is inexact); the
    accumulator count `N` is opt-in via `accumulators` (a no-op below 2). Composes
    after reassociate-reductions, which first folds an unrolled reduction chain
    into a single carried operator.
  }];
  let options = [
    Option<"accumulators", "accumulators", "unsigned", "0",
           "Number of rotated accumulators N (>= operator latency for II=1).">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect"
  ];
}

def SdcSchedulingPass : Pass<"sdc-scheduling", "ModuleOp"> {
  let summary = "Run SDC-based scheduling";
  let description = [{
    Perform SDC-based scheduling on the whole design, and emit the result using
    `allo.sched.*` attributes on operations. Scheduling results can be further
    processed by `convert-schedule-to-dcp` to reify the schedule into `allo.dcp.*`
    ops.
  }];
  let options = [
    Option<"cycleTime", "cycle-time", "float", "0.0",
           "Target clock period (ns); <=0 falls back to the operator library's "
           "cycle_time_ns, else 5.0">,
    Option<"operatorLibrary", "operator-library", "std::string", "\"\"",
           "Path to a YAML operator library; empty uses the built-in default">
  ];
}

#endif // ALLO_TRANSFORMS_PASSES_TD
