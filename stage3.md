# Stage 3: Planning and Code Emission

## Scope

Stage 3 turns a matched, parameter-solved `SemanticGraph` into a flat
`act.sequence` of concrete `act.emit` ops. It is split into an execution planner
and a small emitter.

Implementation files:

- `mlir/include/act/Support/Planning.h`
- `mlir/lib/act/Support/Planning.cpp`
- `mlir/include/act/Support/CodeEmission.h`
- `mlir/lib/act/Support/CodeEmission.cpp`

The current pass driver validates that each lowered function is flat: exactly one
block, no control flow, and no symbol users. Region-local lowering through loops
or branches is unsupported. It is only a possible extension if later workloads
need it, not an assumed future direction.

## Inputs and Output

Inputs:

- `SemanticGraph` from Stage 1.
- `GraphParamSolution` from Stage 2.
- The module's `act.buffer` declarations and identity movement instructions.

Output:

- A new `act.sequence @name` containing scheduled `act.emit` ops.
- The original `func.func @name` is erased.

All emitted addr and compute params are static dense integer attributes. Dynamic
addr SSA operands and extra compute params are not emitted by the current code.

## ExecutionPlan Overview

`buildExecutionPlan(graph, solutions)` constructs:

- `PlanNode`s for selected compute instructions.
- `PlanValue`s for function inputs, placeholders, and produced instruction
  outputs.
- HBM allocations for tensor function boundaries.
- Scratch placements for on-chip buffers.
- Compute and writeback actions.
- Movement nodes and a final flat schedule.

The final schedule contains two kinds of steps:

- `Compute`: emit the selected instruction.
- `Move`: emit an identity movement instruction discovered from `act.define`.

## HBM Mapping

The planner identifies the single declared `!act.hbm` buffer and uses it as one
flat address space.

Sequential allocations are assigned for:

- Ranked tensor function arguments.
- Ranked tensor function results.

Each allocation records:

- Flat base offset.
- Static tensor shape.
- Row-major strides.
- Boundary index.

Dynamic HBM boundary shapes are rejected. HBM capacity is computed from the
declared buffer size and element shape.

Static slices carried by Stage 1 layout chains are converted into flat
row-major regions:

```text
base += slice_offset[dim] * allocation_stride[dim]
stride[dim] = slice_stride[dim] * allocation_stride[dim]
```

This is used for HBM inputs and function-result writebacks.

## Plan Node Construction

For each semantic node and param solution:

1. Copy the selected `DefineOp`.
2. Initialize `paramBindings` with Stage 2 solved shape params.
3. Create one `PlanOperandAccess` per symbolic instruction access.
4. Create one produced `PlanValue` per destination output binding.

Input bindings wire operands to either:

- An HBM input value when the base value is a tensor function argument.
- A produced value from another plan node.
- A placeholder value for external non-argument tensors such as fill/init
  infrastructure.

If a destination operand also appears as a linalg input binding, the operand role
is upgraded from `Write` to `ReadWrite`.

Every read access must be assigned by a binding. Every instruction output must
have a semantic output binding.

## Actions and Lifetimes

The planner creates:

- One compute action per plan node.
- One writeback action per produced value that feeds a function result.

Value lifetimes are the min/max action ids covering:

- The producing compute action.
- All consuming compute actions.
- All writeback actions.

These lifetimes drive scratch reuse and forwarding.

## Scratch Allocation

All non-HBM declared buffers become scratch resources. Current scratch placement
supports only one-dimensional, unit-stride on-chip storage regions.

For each non-HBM read operand:

1. Evaluate the symbolic storage count and stride using current param bindings.
2. Require one storage dimension and stride `1`.
3. Reuse a live placement with the same buffer and size if available.
4. Otherwise allocate a non-overlapping region for this action.
5. Bind the operand's one-dimensional basis expression to the selected offset.

For each non-HBM destination operand:

1. Compute the required storage size.
2. If the destination can alias an input placement whose lifetime ends at this
   action, reuse that placement.
3. Otherwise allocate a placement for the output value's lifetime.
4. Bind the destination basis param to the selected offset.

If a basis expression is already known, it must equal the selected offset. If it
is not a single param or known value, planning fails.

Over-capacity placements and placeholder initialization requirements are recorded
in the plan dump. The current implementation does not synthesize a fix for these
issues.

## Data Movement Catalog

Identity instructions are discovered during planning, not Stage 1. An identity
instruction must:

- Have one source buffer and one destination buffer.
- Have an empty compute region except for `act.yield`.
- Yield only compute block arguments.

The catalog key is `(srcBuffer, dstBuffer)`. Each entry also records a simple
layout signature. The current signature tracks whether the addr chain contains a
transpose and its permutation; expand/collapse are skipped for signature
purposes.

## Movement Planning

For each compute action, required input moves are inserted before the compute
step.

HBM to scratch:

1. Use the operand's HBM region as source.
2. Use the allocated scratch region as destination.
3. Look up a movement instruction from `@hbm` to the destination buffer with the
   required layout signature.

Scratch to scratch:

1. Find a live placement for the input value.
2. Require a different destination buffer and equal size.
3. Look up a direct movement instruction between scratch buffers.

For writeback actions:

- Prefer a direct scratch-to-HBM store.
- If unavailable, try one intermediate scratch buffer where both
  scratch-to-scratch and scratch-to-HBM movement instructions exist.

Movement instructions are currently expected to have exactly three addr params:

```text
addr(src_offset, dst_offset, size)
```

## Emission

`emitInstructionSequence` creates:

```mlir
act.sequence @func_name {
  act.emit @some_inst addr(...) compute()
  ...
}
```

The sequence is inserted after the original function and inherits the function's
symbol name and discardable attributes. Then the original function is erased.

For every scheduled compute or move step:

1. Collect all addr params from the instruction access block argument count.
2. Require every param to have a static binding.
3. Require zero extra compute params.
4. Emit `act.emit` with empty dynamic addr/compute value ranges and dense static
   attrs.

## Current Limitations

- No backend tiling, padding, or loop generation.
- No region-local lowering through affine/scf/control-flow bodies.
- No dynamic addr SSA operands in emitted `act.emit`.
- No extra compute params.
- Direct compute instructions that access HBM with unsolved offset params are
  not handled; offset params must be bound by shape solving, scratch placement,
  or movement endpoint planning before emission.
- Scratch placement only supports 1D, unit-stride on-chip storage regions.
- Movement layout signatures only model transpose.
- Placeholder initialization and over-capacity placements are reported in debug
  dumps but not automatically repaired.
