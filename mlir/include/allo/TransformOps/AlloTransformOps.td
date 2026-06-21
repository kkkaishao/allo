/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORM_OPS
#define ALLO_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "allo/IR/AlloAttrs.td"

def Allo_RaiseToAffineOp : Op<Transform_Dialect, "allo.raise_to_affine",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Raise loop ops to affine form";
  let description = [{
    Input:
      - `$target` is a handle to one or more loop ops.
      - Each payload op must be one of:
        `scf.for`, `scf.parallel`, `affine.for`, or `affine.parallel`.
      - `scf.parallel` with reductions is rejected.

    Behavior:
      - `scf.for` is raised to `affine.for` when bounds/step match affine
        patterns.
      - `scf.parallel` is raised to `affine.parallel` when each dimension's
        bounds/step matches affine patterns.
      - For already-affine loop ops, the loop itself is kept and memref access
        ops inside are raised/canonicalized to affine access ops when possible.

    Output:
      - `$result` contains one payload op per input payload op (same cardinality
        as `$target`).
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    $target attr-dict `:` type($target) `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value":$target), [{
      auto anyType = transform::AnyOpType::get(target.getContext());
      build($_builder, $_state, anyType, target);
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def Allo_OutlineOp : Op<Transform_Dialect, "allo.outline",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Outline payload ops into a callable + callsite";
  let description = [{
    Input:
      - `$target` is a handle to one or more payload operations.
      - Each payload operation must have at most one region.
      - `$kernel_name` is the symbol name used for the outlined callable.
      - `$mapping`, when present, is the dataflow mapping for the outlined
        kernel.

    Behavior:
      - Each target is outlined independently.
      - Without `$mapping`, the outlined callable is a `func.func` and the
        original target site is replaced by a `func.call`.
      - With `$mapping`, the outlined callable is an `allo.kernel` and the
        original target site is replaced by an `allo.invoke`.

    Output:
      - `$kernel` contains created callable ops.
      - `$call` contains created callsite ops.
      - Both result handles have one payload op per input payload op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       StrAttr:$kernel_name,
                       OptionalAttr<DenseI32ArrayAttr>:$mapping);
  let results = (outs TransformHandleTypeInterface:$kernel, TransformHandleTypeInterface:$call);
  let assemblyFormat = [{
    $target `to` $kernel_name (`mapping` `=` $mapping^)? attr-dict
      `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$target, "StringRef":$kernel_name), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      build($_builder, $_state, {anyType, anyType}, target, kernel_name,
            nullptr);
    }]>,
    OpBuilder<(ins "Value":$target, "StringRef":$kernel_name,
                   "ArrayRef<int32_t>":$mapping), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      build($_builder, $_state, {anyType, anyType}, target, kernel_name,
            $_builder.getDenseI32ArrayAttr(mapping));
    }]>
  ];
}

def Allo_LoopReorderOp : Op<Transform_Dialect, "allo.reorder",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Reorder selected loops in an affine perfect band";
  let description = [{
    Input:
      - `$loops` must resolve to at least two `affine.for` ops.
      - The selected loops may be unordered and non-contiguous in the handle.
      - All selected loops must belong to one perfectly nested affine loop band.
      - `$permutation` must be a valid permutation whose length equals the
        number of selected loops.

    Behavior:
      - Reorder is applied to the selected subset.
      - Unselected loops in the same perfect band keep their relative positions.
      - Legality is checked with affine loop interchange constraints.

    Output:
      - No result handle; this op mutates payload IR in place.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loops, DenseI32ArrayAttr:$permutation);
  let results = (outs);
  let assemblyFormat = [{
    $loops `with` $permutation attr-dict `:` type($loops)
  }];
  let hasVerifier = 1;
}

def Allo_LoopSplitOp : Op<Transform_Dialect, "allo.split",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Split each target loop into outer/inner loops";
  let description = [{
    Input:
      - `$loop` is a handle to one or more loop ops.
      - Each payload op must be `affine.for` or `scf.for`.
      - `$factor` must be a positive integer.
      - If loop range is statically known, `$factor` must be strictly smaller
        than the loop trip count.

    Behavior:
      - Each payload loop is split independently (`TransformEachOpTrait`).

    Output:
      - `$outer` contains outer loops, one per input payload loop.
      - `$inner` contains inner loops, one per input payload loop.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loop, I64Attr:$factor);
  let results = (outs TransformHandleTypeInterface:$outer, TransformHandleTypeInterface:$inner);
  let assemblyFormat = [{
    $loop `with` $factor attr-dict `:` functional-type(operands, results)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
  let builders = [
    OpBuilder<(ins "Value":$loop, "int64_t":$factor), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      build($_builder, $_state, anyType, anyType, loop, static_cast<uint64_t>(factor));
    }]>
  ];
}

def Allo_LoopTileOp : Op<Transform_Dialect, "allo.tile",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Tile a loop nest and return tile/point loops";
  let description = [{
    Input:
      - `$loop` is a handle to one or more loops, typically created by
        `transform.merge_handles`.
      - Payload loops must be all `affine.for` or all `scf.for` (mixing
        dialects is rejected).
      - `$factors` must be positive, and its length must match the number of
        payload loops.
      - Payload loops may be unordered; factors are first bound by handle
        iteration order, then loop/factor pairs are sorted by loop depth.
      - Loops must be unique and form one ancestor chain (same loop nest).

    Behavior:
      - Uses perfect or imperfect tiling based on selected-loop contiguity.

    Output:
      - `$tile_loops` and `$point_loops` each contain one loop per selected
        input loop.
      - Both result handles are reported in depth order.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loops, I64ArrayAttr:$factors);
  let results = (outs TransformHandleTypeInterface:$tile_loops, TransformHandleTypeInterface:$point_loops);
  let assemblyFormat = [{
    $loops `with` $factors attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$loops, "ArrayRef<int64_t>":$factors), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      auto factorsAttr = $_builder.getI64ArrayAttr(factors);
      build($_builder, $_state, anyType, anyType, loops, factorsAttr);
    }]>
  ];
}

def Allo_LoopFlattenOp : Op<Transform_Dialect, "allo.flatten",
  [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Flatten the given loop nest";
  let description = [{
    Input:
      - `$loops` must resolve to `affine.for` ops only.
      - Selected loops may be unordered and non-contiguous in the handle.
      - If two or more loops are selected, they must be in one perfectly nested
        affine loop band.

    Behavior:
      - Selected loops are sorted by depth.
      - The op flattens the contiguous perfect band from selected outermost loop
        to selected innermost loop (intermediate loops are included).
      - Flattening requires normalized loops:
        step = 1, constant lower bound = 0, constant positive upper bound.
      - The multiplied trip count must fit in int64.
      - At least two loops must be selected.

    Output:
      - `$result` contains the flattened outermost loop.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loops);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    $loops attr-dict `:` type($loops) `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value":$loops), [{
      auto anyType = transform::AnyOpType::get($_builder.getContext());
      build($_builder, $_state, anyType, loops);
    }]>
  ];
}

def Allo_ComputeAtOp : Op<Transform_Dialect, "allo.compute_at",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Attach a producer loop nest under a consumer affine loop";
  let description = [{
    Input:
      - `$producer` resolves to exactly one payload op.
      - `$consumer_loop` resolves to exactly one `affine.for`.
      - `$producer` must be enclosed by an affine loop nest.

    Behavior:
      - If RAW dependence exists, rely on affine producer-consumer fusion.
      - If no dependence exists, move the producer subtree under
        `$consumer_loop`, with optional `affine.if` guarding for subset bounds.
      - Loop-IV remapping is applied for the common prefix depth.

    Handle effects:
      - Consumes only `$producer`.
      - `$consumer_loop` is read-only and remains reusable.
  }];
  let arguments = (ins TransformHandleTypeInterface:$producer, TransformHandleTypeInterface:$consumer_loop);
  let results = (outs);
  let assemblyFormat = [{
    $producer `at` $consumer_loop attr-dict `:` functional-type(operands, results)
  }];
}

def Allo_ReuseAtOp : Op<Transform_Dialect, "allo.reuse_at",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Create a reuse buffer at a given affine loop axis";
  let description = [{
    Input:
      - `$target` resolves to exactly one memref value, or one op with a
        single memref result.
      - `$axis` resolves to exactly one `affine.for`.

    Behavior:
      - Analyze accesses of `$target` under the loop stage containing `$axis`.
      - Create a reuse buffer and rewrite eligible loads to read from it.

    Output:
      - `$result` is the created reuse buffer handle.
  }];
  let arguments = (ins
    TransformValueHandleTypeInterface:$target,
    TransformHandleTypeInterface:$axis,
    UnitAttr:$use_ring_buffer);
  let results = (outs TransformValueHandleTypeInterface:$result);
  let assemblyFormat = [{
    $target `at` $axis (`ring` $use_ring_buffer^)? attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$target, "Value":$axis, "bool":$use_ring_buffer), [{
      auto anyType = transform::AnyValueType::get($_builder.getContext());
      build($_builder, $_state, anyType, target, axis, use_ring_buffer);
    }]>
  ];
}

def Allo_BufferAtOp : Op<Transform_Dialect, "allo.buffer_at",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Create a buffer at a given affine loop axis";
  let description = [{
    Input:
      - `$target` resolves to exactly one memref value, or one op with a
        single memref result.
      - `$axis` resolves to exactly one `affine.for`.

    Behavior:
      - Create a temporary buffer and rewrite all accesses of `$target` under the loop stage
        containing `$axis` to read/write from it.
      - The original buffer is left unchanged

    Output:
      - Result is the created buffer handle.
  }];
  let arguments = (ins TransformValueHandleTypeInterface:$target, TransformHandleTypeInterface:$axis);
  let results = (outs TransformValueHandleTypeInterface:$result);
  let assemblyFormat = [{
    $target `at` $axis attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$target, "Value":$axis), [{
      auto anyType = transform::AnyValueType::get($_builder.getContext());
      build($_builder, $_state, anyType, target, axis);
    }]>
  ];
}

// name as AlloLoopUnrollOp to avoid naming conflict with existing LoopUnrollOp in other dialects
def Allo_AlloLoopUnrollOp : Op<Transform_Dialect, "allo.unroll",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Unroll the given loops by a constant factor";
  let description = [{
    Input:
      - `$loops` must resolve to loops.
      - `$factor` must be a non-negative integer. 0 means full unroll.

    Behavior:
      - Structurally unroll `scf.for` or `affine.for` loops.

    Output:
      - No result handle; this op mutates payload IR in place.
  }];
  let arguments = (ins TransformHandleTypeInterface:$loops, I64Attr:$factor);
  let results = (outs);
  let assemblyFormat = [{
    $loops `with` $factor attr-dict `:` type($loops)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def Allo_MatchValueOp : Op<Transform_Dialect, "allo.match_value",
  [NavigationTransformOpTrait, MemoryEffectsOpInterface,
   TransformOpInterface, TransformEachOpTrait,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Match target values";
  let description = [{
    Match target values from a target operation.

    - `$number` is the selected index.
    - `$source_kind` controls which value space to index:
      - `0` (`auto`): keep old behavior, prefer block arguments when present,
        otherwise use operation results.
      - `1` (`block_arg`): index only block arguments.
      - `2` (`result`): index only operation results.
  }];
  let arguments = (ins
    TransformHandleTypeInterface:$target,
    I64Attr:$number,
    DefaultValuedAttr<I64Attr, "0">:$source_kind);
  let results = (outs TransformValueHandleTypeInterface:$result);
  let assemblyFormat = [{
    $number `of` $target `kind` $source_kind attr-dict `:` type($target) `->` type($result)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
  let builders = [
    OpBuilder<(ins "Value":$target, "int64_t":$number), [{
      auto anyType = transform::AnyValueType::get($_builder.getContext());
      build($_builder, $_state, anyType, target, static_cast<int64_t>(number), static_cast<int64_t>(0));
    }]>,
    OpBuilder<(ins "Value":$target, "int64_t":$number, "int64_t":$source_kind), [{
      auto anyType = transform::AnyValueType::get($_builder.getContext());
      build($_builder, $_state, anyType, target, static_cast<int64_t>(number), static_cast<int64_t>(source_kind));
    }]>
  ];
}

def Allo_PartitionOp : Op<Transform_Dialect, "allo.partition", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<TransformOpInterface>,
  ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Partition the given array";
  let description = [{
    Input:
      - `$target` must resolve to exactly one memref value, or one op with a
        single memref result.

    Behavior:
      - Attach the given partitioning attribute to the target value
  }];
  let arguments = (ins TransformValueHandleTypeInterface:$target, PartitionAttr:$partition);
  let results = (outs);
  let assemblyFormat = [{
    $target `with` $partition attr-dict `:` type($target)
  }];
}

def Allo_BindStorageOp : Op<Transform_Dialect, "allo.bind_storage", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<TransformOpInterface>,
  ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Bind the given array to a specific memory resource";
  let description = [{
    Input:
      - `$target` must resolve to exactly one memref value, or one op with a
        single memref result.
      - `$mem_type` is the Vitis storage type (e.g. `RAM_2P`, `ROM_1P`).
      - `$impl` is the storage implementation (e.g. `BRAM`, `URAM`, `LUTRAM`).

    Behavior:
      - Attach an `allo.bind.storage` dictionary attribute (`type`/`impl`) to the
        target buffer's root carrier (function argument, alloc, or global). The
        Vivado HLS emitter turns it into `#pragma HLS bind_storage`.
  }];
  let arguments = (ins TransformValueHandleTypeInterface:$target,
                       StrAttr:$mem_type, StrAttr:$impl);
  let results = (outs);
  let assemblyFormat = [{
    $target `type` $mem_type `impl` $impl attr-dict `:` type($target)
  }];
}

def Allo_StreamlineOp : Op<Transform_Dialect, "allo.streamline",
  [AttrSizedOperandSegments,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<TransformOpInterface>,
   ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Convert the memory boundaries between kernels into streams";
  let description = [{
    Input:
      - `$producers` and `$consumers` are `allo.invoke` handles in a common
        parent region (a fused `top`). Each invoke's callee is resolved to its
        (unique) kernel; pointing at the call sites rather than the kernels makes
        the boundary -- the shared operand -- explicit.

    Behavior:
      - Discover every *memory boundary*: a memref the producer(s) only write and
        the consumer(s) only read -- a DRAM intermediate handed between them.
      - For each boundary, retype the matching argument on each side from
        `memref` to `!allo.stream`, staging through an on-chip buffer only where a
        side's access isn't whole-tensor row-major, and replace the shared memref
        with a `stream.create` so the kernels hand off over a FIFO instead of DRAM:
        * one producer, one consumer -- a direct hand-off;
        * one producer, many consumers (fan-out) -- the producer writes one FIFO
          and a generated `tee` kernel broadcasts it to one FIFO per consumer
          (residual / skip connections);
        * many producers, one consumer (fan-in) -- each producer writes a disjoint
          contiguous row-major block to its own FIFO and a generated `merge`
          kernel concatenates them in order into the consumer's FIFO.
      - Boundaries that cannot be converted are skipped with a diagnostic
        (argument unused, read-write aliasing, non-affine / non-load-store uses, a
        non-static footprint, a fan-in producer that isn't a contiguous block, or
        an unsupported many-to-many boundary). Other boundaries are still done.
      - `$depth` sets the FIFO depth of every stream created (default 2). On a
        *reconvergent* fork/join (e.g. a residual: one value reaching a join both
        directly and through a longer path), the short path's FIFO must hold the
        latency skew or the dataflow deadlocks; a too-shallow depth there is
        flagged with a warning naming the fork and join so the depth can be set.

    Output:
      - No result handle; this op mutates payload IR in place. The `$producers`
        and `$consumers` handles remain valid (still point at the same kernels).
  }];
  let arguments = (ins Variadic<TransformHandleTypeInterface>:$producers,
                       Variadic<TransformHandleTypeInterface>:$consumers,
                       DefaultValuedAttr<I64Attr, "1">:$lanes,
                       DefaultValuedAttr<I64Attr, "2">:$depth);
  let results = (outs);
  let assemblyFormat = [{
    `(` $producers `)` `(` $consumers `)` attr-dict
    `:` `(` type($producers) `)` `(` type($consumers) `)`
  }];
}

#endif // ALLO_TRANSFORM_OPS
