# Stage 3: Code Emission — Detailed Design

## Status

**Iteration 1 (direct-buffer, flat emission): PLANNED**
**Iteration 2 (data movement, multi-buffer): PLANNED**
**Iteration 3 (multi-op functions, affine offset tracing): PLANNED**

Previous code emission (generating `scf.for` tiling loops, dynamic HBM offset computation from loop IVs) has been removed. The backend no longer generates tiling loops — it assumes the input is already well-tiled. Code emission is now simpler: flat sequences of `act.emit` calls per region, with offsets derived from the source IR.

## Goal

For each matched and parameter-solved node, emit the final `act.emit` IR: data movement instructions and compute instructions, with addr params derived from solved shape params and source-IR offset information.

## Architectural Changes from Previous Design

| Concern | Previous | New |
|---------|----------|-----|
| Loop generation | Backend generates `scf.for` from tiling factors | No loops generated — input already has affine loops if needed |
| Offset computation | Dynamic offsets from tiling loop IVs | Offsets traced from source IR using `AffineValueMapBuilder` |
| Accumulator init | Backend emits pre-loop loads | Midend responsibility (e.g., `linalg.fill` before loop) |
| Reduction ordering | Backend separates parallel/reduction loops | Not applicable — no loop generation |
| Scope of emission | Per-function | Per-region (handles ops inside affine loop bodies) |

### Key design principle: per-region instruction selection

The backend operates **per-region**. When the source IR contains affine loop nests (from midend tiling), the backend runs instruction selection inside each innermost loop body independently. This is sufficient because:

1. Affine loops are analyzable — the `AffineValueMapBuilder` can compose offset expressions through the loop structure.
2. Each innermost body contains the tiled linalg ops that fit instruction capacity.
3. The backend doesn't need to understand the loop structure — it just matches and emits within each region.

### Offset tracing via AffineValueMapBuilder

The `AffineValueMapBuilder` (copied from `allo/TransformOps/Utils.cpp` into `act/Support/`) traces SSA operand chains and composes them into affine maps. This replaces the previous ad-hoc loop-IV-based offset computation.

For a source operand that comes from a `tensor.extract_slice` (or affine load), the builder:
1. Imports the slice's offset operands as affine expressions
2. Recursively traces through `arith.addi`, `arith.muli`, `arith.divsi`, `arith.remsi`, `affine.apply`
3. Composes into a single `AffineMap` with loop IVs as dimensions and constants as symbols
4. The resulting map can be emitted as `affine.apply` or evaluated statically if all operands are constant

## Input / Output

- **Input**: `SemanticsGraphs` from Stage 1, `GraphParamSolution` from Stage 2, plus the source module with `act.define`, `act.buffer`, and source `func.func` ops.
- **Output**: Transformed module where source linalg ops are replaced by `act.emit` ops. Source functions become buffer-level programs (no tensor args/returns).

---

## Architecture: Three Phases

### Phase 3a: Logical planning
Build a `LogicalPlan` from the semantics graph and param solutions: map matched nodes to logical values, wire inter-node dataflow, record transform chains.

### Phase 3b: Resource planning
Assign HBM regions for source tensors, allocate scratchpad buffer slots for instruction operands, build the data movement catalog, detect forwarding opportunities.

### Phase 3c: Code emission
Emit `act.emit` calls with computed addr params, data movement instructions, and cleanup.

---

## Phase 3a: Logical Planning

### Goal
Convert the `SemanticsGraph` (matched nodes + edges) into a `LogicalPlan` that carries:
- Logical tensor values flowing between compute nodes
- Per-node input/output bindings with transform chains
- Writeback targets for produced values

### Algorithm

1. **Register function arguments** as `FunctionInput` logical values.
2. **Create planner nodes** for each `SemanticsGraphNode`:
   - Record source ops, instruction, and param solution
   - Create `Produced` logical values for each output
   - Wire writeback targets from boundary output bindings
3. **Wire inter-node edges** from `SemanticsGraph.edges`:
   - Map producer output values to consumer input slots
   - Carry transform chains from the graph edges
4. **Wire external inputs** from boundary input bindings
5. **Validate** that every tensor operand at the compute boundary has been assigned

### Data structures

```cpp
struct LogicalPlanNode {
  SmallVector<Operation *, 4> sourceOps;
  DefineOp instruction;
  const ParamSolution *paramSolution;  // from Stage 2
  SmallVector<LogicalPlanNodeInput, 2> inputs;
  SmallVector<LogicalPlanNodeOutput, 1> outputs;
};

struct LogicalPlan {
  SmallVector<LogicalPlanNode, 4> nodes;
  SmallVector<LogicalPlanValue, 8> values;
  DenseMap<Value, unsigned> externalValueIds;
};
```

---

## Phase 3b: Resource Planning

### HBM layout

Map source tensor values to HBM regions. Identify the HBM buffer via `!act.hbm<...>` type.

**Layout assignment** (unchanged from previous design):
```
offset = 0
for each logical value (live range analysis + greedy allocation):
  hbmLayout[valueId] = { offset, shape, strides }
  strides = row-major strides
  offset += product(shape)
```

Layout aliases are computed for produced values whose writeback target is physically identical.

### Scratchpad allocation

For multi-buffer instructions, allocate slots in scratchpad buffers:

```
for each planned node:
  for each operand:
    buffer = declared buffer for this operand
    slotsNeeded = evaluateOperandSlotCount(instruction, operandIdx, solvedParams)
    allocate sequential slots in buffer
```

`evaluateOperandSlotCount` uses `ParamSolution::solvedParams` to evaluate the symbolic count expressions from the StridedOp.

### Data movement catalog

Build a catalog mapping (srcBuffer, dstBuffer) to identity `DefineOp`s, with layout signatures (transpose detection). Unchanged from previous design.

### Forwarding detection

Detect scratchpad forwarding opportunities between adjacent nodes — when a producer's output buffer slot can be directly consumed by the next node without an HBM round-trip. Unchanged from previous design, but simpler without tiling loop considerations.

---

## Phase 3c: Code Emission

### Overview

For each planned node, in topological order within each region:

1. **Compute addr params**:
   - Shape params: from `ParamSolution::solvedParams`
   - Offset params: from HBM layout (static) or traced from source IR (dynamic)
2. **Emit data movement** (if multi-buffer):
   - Load instructions: HBM -> scratchpad for each src operand
3. **Emit compute**: `act.emit @instruction addr(...) compute(...)`
4. **Emit stores** (if multi-buffer):
   - Store instructions: scratchpad -> HBM for each dst operand
5. **Cleanup**: erase source ops, dead infrastructure, update function signature

### Offset computation

#### Static offsets (no enclosing loops)

When a source op has no enclosing affine loops, all offsets are static:

```
for each offset param p:
  operandIdx = paramToOperand[p]
  hbmValueId = getHBMValueForOperand(node, operandIdx)
  offset = hbmLayout[hbmValueId].baseOffset
  // Apply logical transforms (transpose, slice) to the layout
  transformedOffset = applyLogicalTransforms(layout, transforms)
  staticAddrParams[p] = transformedOffset
```

#### Dynamic offsets (inside affine loops)

When a source op is inside an affine loop nest (from midend tiling), offsets depend on loop induction variables.

**Approach: Trace offsets from source IR using `AffineValueMapBuilder`.**

The source IR after midend tiling looks like:
```mlir
affine.for %i = 0 to 4 {
  affine.for %j = 0 to 4 {
    %slice_a = tensor.extract_slice %A[%i*64, %j*64] [64, 64] [1, 1]
    %result = linalg.matmul ins(%slice_a, ...) outs(...)
    ...
  }
}
```

For operand `%slice_a`, the offset into tensor `%A` is `%i*64 + %j*64*stride`. The `AffineValueMapBuilder` traces the slice's offset operands through the SSA chain and composes them into an affine map `(d0, d1) -> (d0 * 64 * stride_row + d1 * 64)` where `d0 = %i`, `d1 = %j`.

**Steps:**
1. For each offset param, identify the source operand and its defining op (e.g., `tensor.extract_slice`)
2. Use `AffineValueMapBuilder` to import the slice's offset operands
3. Compose to get an `AffineMap` from loop IVs to flat HBM offset
4. Add HBM base offset from the resource plan
5. Emit the offset as `affine.apply` of the composed map, or as a static constant if it simplifies

This naturally handles:
- Static offsets (constant map)
- Simple stride patterns (single loop with constant step)
- Nested loops with non-trivial addressing
- Compositions through `arith.addi`, `arith.muli`, `affine.apply` chains

### Data movement emission

#### Load emission
For each src operand not already in the correct buffer:
1. Look up identity instruction in data movement catalog: `(HBM_buffer, src_buffer)`
2. Match layout signature (transpose if needed)
3. Compute HBM offset (static or dynamic as above)
4. Compute scratchpad slot offset from operand residence
5. Emit `act.emit @load addr(hbm_offset, spad_slot, size) compute()`

#### Store emission
For each dst operand not writing directly to HBM:
1. Look up identity instruction: `(dst_buffer, HBM_buffer)`
2. Compute offsets
3. Emit `act.emit @store addr(spad_slot, hbm_offset, size) compute()`

If a direct path doesn't exist, try multi-hop (e.g., d2 -> d1 -> HBM via `@mov_back` + `@store_rm`).

#### Dynamic data movement
When offsets are dynamic (inside loops), use `emitDataMovementDynamic` which accepts SSA values for the dynamic offset while keeping other params static.

### Function signature transformation

After emission, rewrite the function:
- Remove tensor arguments (data is at assigned HBM offsets)
- Remove tensor returns
- Function becomes `func.func @name() { ... act.emit ... return }`

### Cleanup

1. Erase source compute ops (in reverse topological order)
2. Iteratively erase dead infrastructure: `linalg.fill`, `tensor.empty`, `arith.constant`, layout ops
3. Preserve `arith.constant` ops for index values created during emission

---

## Implementation Plan

### Iteration 1: Direct-buffer, flat emission

**Goal:** Handle instructions where all operands are in the same buffer (e.g., tpu.mlir's `@matmul` on `@devmem`). No data movement, no enclosing loops. Single compute op per function.

#### Step 1: Logical planning (Phase 3a)
- Build `LogicalPlan` from `SemanticsGraph` + `GraphParamSolution`
- Replace `TilingAnalysis*` with `ParamSolution*` in `LogicalPlanNode`
- Wire values and validate completeness

#### Step 2: Resource planning (Phase 3b)
- HBM layout: sequential allocation with live range analysis
- No scratchpad needed (single-buffer)
- No data movement catalog needed

#### Step 3: Flat code emission (Phase 3c)
- For each node: compute static addr params (shape from solved params, offset from HBM layout)
- Emit single `act.emit` per node
- Cleanup source ops and function signature

**Test:** `square_mm.mlir` (8x8) + tpu.mlir -> single `act.emit @matmul addr(0, 64, 128, 8) compute()`

### Iteration 2: Data movement, multi-buffer

**Goal:** Handle multi-buffer instructions (e.g., qkv.mlir's `@gemm` with d0/d1/d2 buffers). Emit load/compute/store sequences.

#### Step 4: Data movement catalog
- Identify identity instructions
- Build (srcBuffer, dstBuffer) -> DefineOp map with layout signatures

#### Step 5: Scratchpad allocation
- Per-buffer sequential slot allocation using `evaluateOperandSlotCount` with `ParamSolution::solvedParams`

#### Step 6: Load/store emission
- Emit load instructions before compute
- Emit store instructions after compute
- Handle multi-hop buffer transitions (e.g., d2 -> d1 -> HBM)
- Forwarding detection for adjacent nodes

**Test:** `mm_bf16.mlir` (64x64 bf16) + qkv.mlir -> `@load_rm, @load_rm, @gemm, @mov_back, @store_rm`

### Iteration 3: Multi-op functions and affine offset tracing

**Goal:** Handle functions with multiple compute ops, intermediate results, and ops inside affine loop bodies.

#### Step 7: Copy AffineValueMapBuilder into act/Support/
- Copy from `allo/TransformOps/Utils.cpp` into `act/Support/AffineMapComposer.{h,cpp}`
- Remove allo-specific dependencies (`allo/IR/AlloOps.h`)
- Keep the core: `AffineValueMapBuilder`, `importValue`, `compose`, arithmetic tracing

#### Step 8: Per-region instruction selection
- Walk function body; for ops inside affine loop bodies, run matching + emission within that region
- Identify enclosing affine loop IVs for dynamic offset computation

#### Step 9: Affine offset tracing
- For source operands defined by `tensor.extract_slice` or similar, use `AffineValueMapBuilder` to compose offset expressions
- Emit `affine.apply` for dynamic offsets, or static constants when map is constant
- Add HBM base offset from resource plan

#### Step 10: Multi-op intermediate results
- HBM layout for intermediate tensor values
- Scratchpad reuse across nodes
- Correct handling of CSE'd DPS init values

**Test:** `chain_mm.mlir` (two sequential matmuls) + tpu.mlir -> two `act.emit @matmul` with correct HBM offsets

---

## Worked Examples

### Example 1: Single matmul, direct buffer

**Source:**
```mlir
func.func @mm(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = linalg.matmul ins(%a, %b) outs(%fill) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

**ISA:** tpu.mlir with `@matmul` (src(@devmem, @devmem) dst(@devmem), parametric %size)

**Stage 2:** solvedParams = {3: 8}, paramKinds = {0: Offset, 1: Offset, 2: Offset, 3: Shape}

**Emission:** Single `act.emit @matmul addr(0, 64, 128, 8) compute()`
- p0 (offset, A) = 0 (HBM base of %a)
- p1 (offset, B) = 64 (HBM base of %b)
- p2 (offset, C) = 128 (HBM base of %0)
- p3 (shape) = 8 (solved)

**Output:**
```mlir
func.func @mm() {
  act.emit @matmul addr(0, 64, 128, 8) compute()
  return
}
```

### Example 2: Multi-buffer matmul with data movement

**Source:**
```mlir
func.func @mm(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %cst = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<64x64xbf16>
  %fill = linalg.fill ins(%cst : bf16) outs(%empty : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %0 = linalg.matmul ins(%a, %b) outs(%fill) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}
```

**ISA:** qkv.mlir with `@gemm` (src(@d1, @d1) dst(@d2), fixed 64x64)

**Stage 2:** solvedParams = {} (all constant), all offset params

**Emission:**
```mlir
func.func @mm() {
  act.emit @load_rm addr(0, 0, 64) compute()      // A: d0[0] -> d1[0]
  act.emit @load_rm addr(4096, 64, 64) compute()  // B: d0[4096] -> d1[64]
  act.emit @gemm addr(0, 64, 0) compute()         // d1[0] x d1[64] -> d2[0]
  act.emit @mov_back addr(0, 0, 64) compute()     // d2[0] -> d1[0]
  act.emit @store_rm addr(0, 8192, 64) compute()  // d1[0] -> d0[8192]
  return
}
```

### Example 3: Tiled matmul (midend-tiled input, inside affine loops)

**Source (after midend tiling):**
```mlir
func.func @mm(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // ... fill ...
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 2 {
      affine.for %k = 0 to 2 {
        %a_tile = tensor.extract_slice %A[%i*64, %k*64] [64, 64] [1, 1]
        %b_tile = tensor.extract_slice %B[%k*64, %j*64] [64, 64] [1, 1]
        // ... extract C tile, compute, insert back ...
        %c = linalg.matmul ins(%a_tile, %b_tile) outs(%c_tile)
        // ... insert_slice result back ...
      }
    }
  }
  return %result
}
```

**Backend sees:** Inside the innermost affine loop body, there's a `linalg.matmul` on 64x64 tensors.

**Stage 1:** Matches `linalg.matmul` -> `@matmul` (or `@gemm`)

**Stage 2:** Shapes fit (64x64 == instruction native). solvedParams resolved.

**Stage 3 offset tracing:** For operand `%a_tile` from `tensor.extract_slice`:
- `AffineValueMapBuilder` traces offsets `[%i*64, %k*64]`
- Composes with row-major strides of `%A` (shape [128,128], strides [128,1])
- Result: flat offset = `%i * 64 * 128 + %k * 64` = affine map `(d0, d1) -> (d0 * 8192 + d1 * 64)`
- Add HBM base offset of `%A`
- Emit `affine.apply` for the dynamic offset

**Output (inside the affine loop body):**
```mlir
%a_off = affine.apply affine_map<(d0, d1) -> (d0 * 8192 + d1 * 64)>(%i, %k)
%b_off = affine.apply affine_map<(d0, d1) -> (16384 + d0 * 8192 + d1 * 64)>(%k, %j)
%c_off = affine.apply affine_map<(d0, d1) -> (32768 + d0 * 8192 + d1 * 64)>(%i, %j)
act.emit @matmul addr(%a_off, %b_off, %c_off, 64) compute()
```

---

## Edge Cases

### Padding
If a source op's shape doesn't exactly divide the instruction's native shape, the match is **rejected** by Stage 2. Padding is a midend concern — the midend should pad tensors before tiling if needed.

### Fill absorption
`linalg.fill` is not a compute instruction. If it appears inside a tiled loop (for accumulator initialization), it should be matched to a fill/zero instruction in the ISA, or handled as part of midend lowering. The backend treats it as any other matchable op.

### Dynamic shapes
The MVP requires all shapes to be statically known. Dynamic shapes would require runtime parameter solving. Deferred.
