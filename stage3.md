# Stage 3: Coverage Resolution and Code Emission — Detailed Design

## Status

**Iteration 1 (direct-buffer, single compute op): COMPLETE (2026-04-13)**

Implemented files:
- `mlir/include/act/Support/CodeEmission.h`: data structures (`InstructionPlan`, `TensorLayout`, `MemoryLayout`) + `runCodeEmission` declaration
- `mlir/lib/act/Support/CodeEmission.cpp`: phases 3a (coverage resolution), 3b (memory layout), 3c (code emission with loop nest + act.emit + cleanup)
- `mlir/lib/act/Support/CMakeLists.txt`: added CodeEmission.cpp + MLIRSCFDialect
- `mlir/lib/act/Conversion/ConvertCanonicalFormToAct.cpp`: wired Stage 3 after Stage 2

Test results:
- `batch_mm.mlir` (4×8×8) + tpu.mlir → batch loop with `act.emit @matmul addr(%a, %b, %c, 8) compute()`
- `square_mm.mlir` (8×8) + tpu.mlir → single `act.emit @matmul addr(0, 64, 128, 8) compute()` (no loops)
- `mm.mlir` (10×20×30 non-square) + tpu.mlir → correctly rejected: "no valid instruction match"

**Iteration 2 (data movement, multi-buffer): COMPLETE (2026-04-13)**

New/extended:
- `mlir/include/act/Support/CodeEmission.h`: added `DataMovementCatalog`, `ScratchpadSlot`, `ScratchpadLayout` structs; `MemoryLayout` extended with `needsDataMovement`, scratchpad, and catalog fields
- `mlir/lib/act/Support/CodeEmission.cpp`: identity instruction detection, data movement catalog builder, HBM buffer identification, scratchpad slot allocation, load/store/mov emission helpers, multi-buffer code emission path
- `drafts/isa/qkv.mlir`: added `@mov_back` (d2->d1) identity instruction
- `drafts/models/mm_bf16.mlir`: new test model (64×64 bf16 matmul)

Key features:
- Automatic identity instruction detection (compute body yields block args)
- Data movement catalog: maps (srcBuffer, dstBuffer) → DefineOp, prefers non-transposed variants
- HBM buffer identification via `!act.hbm<...>` type
- Scratchpad slot allocation: sequential per-buffer, evaluated from StridedOp counts using SymExpr
- Buffer transition detection: finds multi-hop paths (e.g., d2→d1→d0 via @mov_back + @store_rm)
- Backward compatible: single-buffer instructions (tpu.mlir) still use Iteration 1 path

Test results:
- `mm_bf16.mlir` (64×64 bf16) + qkv.mlir → `@load_rm, @load_rm, @gemm, @mov_back, @store_rm` sequence
- `batch_mm.mlir` (4×8×8) + tpu.mlir → unchanged (regression pass)

**Iteration 3 (multi-op functions): COMPLETE (2026-04-13)**

Changes:
- `mlir/lib/act/Support/CodeEmission.cpp`:
  - Per-entry scratchpad allocation with reuse (each entry allocates from slot 0, max tracked for capacity check) — fixes buffer overflow when multiple compute ops share the same on-chip buffers
  - HBM layout assigned to compute op results instead of DPS inits — fixes incorrect offset sharing when CSE folds identical `linalg.fill` ops
  - `getSourceTensorForOperand`: dst operands now return the op's result Value instead of `getDpsInits()` — ensures correct HBM offset lookup for output operands
- `drafts/models/chain_mm.mlir`: new test (two sequential 8×8 f32 matmuls)
- `drafts/models/chain_mm_bf16.mlir`: new test (two sequential 64×64 bf16 matmuls, reuses input A)

Key features:
- Multiple compute ops per function with intermediate results flowing through HBM
- Scratchpad reuse across entries (each entry stores to HBM before next loads)
- Correct handling of CSE'd DPS init values (layout by results, not by inits)
- Backward compatible: single-op functions unchanged

Test results:
- `chain_mm.mlir` (8×8, two matmuls) + tpu.mlir → `@matmul addr(0,64,192,8), @matmul addr(192,128,256,8)`
- `chain_mm_bf16.mlir` (64×64, two matmuls) + qkv.mlir → two load/gemm/mov_back/store sequences with correct HBM offsets (8192, 12288) and scratchpad reuse
- All Iteration 1/2 tests pass unchanged (regression)

## Goal

Select a non-overlapping instruction for every source compute op and emit the final IR: `act.emit` calls inside `scf.for` tiling loops, with data movement instructions bridging the storage hierarchy.

## Input / Output

- **Input**: `SmallVector<TiledMatchCandidate>` from Stage 2, plus the source module with `act.define`, `act.buffer`, and source `func.func` ops.
- **Output**: Transformed module where source linalg ops are replaced by `act.emit` ops inside `scf.for` loops. Source functions become buffer-level programs (no tensor args/returns).

---

## Architecture: Three Phases

### Phase 3a: Coverage resolution
Select the best valid `TiledMatchCandidate` per source compute op.

### Phase 3b: Memory layout
Assign HBM regions for source tensors and scratchpad buffer slots for instruction operands.

### Phase 3c: Code emission
Generate loop nests, data movement, and compute `act.emit` calls.

---

## Phase 3a: Coverage Resolution

### Algorithm

For each source linalg compute op (not `linalg.fill`, `tensor.empty`, or other infrastructure):
1. Collect all valid `TiledMatchCandidate`s for this op (`isValid == true`)
2. If zero: emit diagnostic "no instruction found for <op> at <loc>" — compilation failure
3. If one: select it
4. If multiple: rank by preference:
   - Fewer total tiling iterations (product of all tile factors)
   - Fewer tiling dimensions with tileFactor > 1
   - Tie-break: first match

### Output

```cpp
struct InstructionPlan {
  /// One entry per source compute op, in topological order.
  struct Entry {
    Operation *sourceOp;
    const TiledMatchCandidate *match;
  };
  SmallVector<Entry> entries;
};
```

### Implementation notes

- Walk source ops inside `func.func` (excluding those inside `act.define`)
- Skip non-compute ops: `linalg.fill`, `tensor.empty`, `arith.constant`, `tensor.pad`
- A source op may have matches from both semantic matching (Stage 1) and structural matching (Stage 2 iteration 2). Stage 2 already unified them — just filter by `isValid`.

---

## Phase 3b: Memory Layout

### HBM layout

Map source tensor values to HBM regions. HBM is the "external memory" buffer — typically the first `act.buffer` with `!act.hbm` type, or the largest scalar buffer (like `@devmem`).

**Identifying the HBM buffer**: Walk `act.buffer` declarations. A buffer is "HBM-like" if its type is `!act.hbm<...>` or if it's the buffer used as `src` in load-type identity instructions. For the MVP, accept a single HBM buffer per ISA (error if ambiguous).

For instructions that operate directly on HBM (like tpu.mlir's `@matmul` with `src(@devmem) dst(@devmem)` where `@devmem` is `!act.scalar<f32>` with size 8192): treat `@devmem` as the single shared memory. No separate HBM concept — everything is in `@devmem`.

**Layout assignment**:

```
offset = 0
for each tensor value in the function (args + intermediates + returns):
  hbmLayout[value] = { offset, shape, strides }
  strides = row-major strides for the shape
  offset += product(shape)  // in buffer elements
```

For the MVP: no reuse of HBM space for dead values. Check that total allocation fits within the buffer's declared size.

> **Note on element types**: The HBM offset is in units of the buffer's element type. For `!act.scalar<f32>`, one element = one f32. For `!act.hbm<16384xbf16>`, one element = one bf16. If the source tensor element type differs from the buffer element type, a conversion is needed (out of MVP scope — for now, require matching element types).

### Scratchpad buffer allocation

For each instruction in the plan, allocate slots in the scratchpad buffers referenced by `src(...)` and `dst(...)`.

**Simple sequential allocator per buffer**:

```
// Per-buffer state:
nextSlot: DenseMap<StringRef, int64_t>  // buffer name → next free slot

for each instruction in plan:
  for each operand (src_0, ..., src_n, dst_0, ..., dst_m):
    bufferName = declared buffer for this operand
    slotsNeeded = tiles consumed by one invocation
    slot = nextSlot[bufferName]
    nextSlot[bufferName] += slotsNeeded
    operandSlot[operand] = slot
```

The "slots needed" for an operand is determined by the instruction's access pattern. For `act.strided basis(?) counts(C) strides(S)` with `C` evaluated using solved shape params, the number of slots = evaluated C value.

For the MVP: allocate once at plan construction time. All tiling loop iterations reuse the same slots (scratchpad is a temporary workspace rewritten each iteration). Check total allocation fits within each buffer's declared size.

### Data movement catalog

Build a catalog of identity instructions for data movement:

```cpp
struct DataMovementCatalog {
  /// Map: (srcBuffer, dstBuffer) → DefineOp for identity instructions
  DenseMap<std::pair<StringAttr, StringAttr>, DefineOp> catalog;
};
```

An identity instruction is one whose compute region is just `act.yield %input` — it copies data between buffers without computation. Detected by checking that the compute region has exactly one non-yield, non-constant op, and it's the yield of a block argument. Or more precisely: the compute body's terminator yields exactly its block arguments (possibly a subset).

Walk all `act.define` ops and identify identity instructions by checking if the compute region is trivially `yield %blockArg`.

---

## Phase 3c: Code Emission

### Overview

For each source compute op in topological order:

1. Determine the loop nest structure from the tiling scheme
2. Emit `scf.for` loops
3. Inside the innermost loop body:
   a. Compute HBM offsets for each operand tile (function of loop IVs)
   b. Emit load instructions (HBM → scratchpad)
   c. Emit compute instruction
   d. Emit store instructions (scratchpad → HBM)
4. Handle reduction dimensions: accumulator initialization + store placement
5. Erase the original source op

### Loop nest structure

Given a `TilingScheme` with per-dimension info:

```
dims[i]: { sourceBound, nativeValue, tileFactor, iterType }
```

Only dims with `tileFactor > 1` generate loops. Ordering:
1. **Parallel dims** (outer): batch dims first (numOuterDims), then remaining parallel dims
2. **Reduction dims** (inner): tiled reduction dims

```
// Pseudocode
for each parallel dim with tileFactor > 1:
  scf.for %iv_p = 0 to tileFactor step 1:
    for each reduction dim with tileFactor > 1:
      scf.for %iv_r = 0 to tileFactor step 1:
        // emit load + compute
    // emit store (outside reduction loops, inside parallel loops)
```

Reduction loop placement matters for correctness:
- The accumulator (output) must be initialized to zero **before** the reduction loop
- The accumulator is updated in-place by each compute invocation across the reduction loop
- The store to HBM happens **after** the reduction loop completes

### Computing HBM tile offsets

For a tiled invocation at loop iteration `(iv_0, iv_1, ..., iv_{n-1})`:

The tile starts at iteration space point `(iv_0 * native_0, iv_1 * native_1, ...)`.

For source operand `k` with indexing map `M_k: (d_0, ..., d_{n-1}) → (e_0, ..., e_{r-1})`:

The operand tile starts at tensor position:
```
tensorOffset[j] = M_k(iv_0 * native_0, ..., iv_{n-1} * native_{n-1})[j]
```

Since linalg indexing maps are affine projections (typically `AffineDimExpr`s), this simplifies to:
```
tensorOffset[j] = iv_{src(j)} * native_{src(j)}
```
where `src(j)` is the iteration dimension that maps to operand dimension `j`.

The HBM element offset (row-major linearization):
```
hbmOffset = hbmBase + sum(tensorOffset[j] * stride[j] for j in 0..r-1)
```
where `hbmBase` is the HBM base offset for this tensor, and `stride[j]` are row-major strides.

For rank-mismatched ops with `numOuterDims > 0`: the outer dims are treated identically — they're just additional parallel loop dimensions whose iteration offsets map to the batch dimensions of the source tensor.

### Emit data movement instructions

**Load emission**: For each `src` operand of the compute instruction:

1. Determine which buffer the operand reads from (from `act.define`'s `src(...)` list)
2. If the data is already in that buffer: skip (for instructions that operate directly on HBM, like `@matmul` on `@devmem`)
3. Otherwise, look up the identity instruction for `(HBM_buffer, src_buffer)` in the data movement catalog
4. Compute addr params for the load instruction:
   - The "input" addr param = HBM tile offset (computed above)
   - The "output" addr param = allocated scratchpad slot
   - Shape params = from the compute instruction's solved params (or tile size)
5. Emit `act.emit @load_instr addr(hbm_offset, scratchpad_slot, shape_params...) compute()`

**Store emission**: For each `dst` operand:
1. Determine the destination buffer from `act.define`'s `dst(...)` list
2. If the instruction writes directly to HBM: skip
3. Otherwise, look up the identity instruction for `(dst_buffer, HBM_buffer)`
4. Compute addr params similarly (reversed: scratchpad → HBM)
5. Emit `act.emit @store_instr addr(scratchpad_slot, hbm_offset, shape_params...) compute()`
6. Place stores **outside** the reduction loop (only store the final accumulated result)

**Internal transfers**: If a compute instruction's output buffer differs from the next instruction's input buffer (e.g., `@gemm` writes to `@d2` but `@store_rm` reads from `@d1`), insert a `@mov`-type instruction between them. Look up identity instruction for `(compute_dst_buffer, store_src_buffer)`.

### Emit compute instruction

Compute the addr params for the compute instruction:
- **Offset params** (classified as `Offset` or `Mixed` by Phase 2b): the allocated scratchpad slot offsets
- **Shape params** (classified as `Shape` by Phase 2b): the solved param values from the tiling scheme

Emit: `act.emit @instr addr(offset_0, offset_1, ..., shape_0, ...) compute()`

The ordering of addr params must match the `act.define`'s addr block arguments. The param classification from Phase 2b tells us which block arg index maps to which kind of param.

### Accumulator handling (reduction dims)

When the tiling scheme has reduction dims with tileFactor > 1:

1. **Initialize**: Before the reduction loop, zero out the accumulator's scratchpad region. This can be done by:
   - Emitting a `linalg.fill` on the buffer region (lowered later), or
   - Emitting an identity instruction that writes zeros, or
   - For the MVP: assume the compute instruction handles accumulation correctly (i.e., `linalg.matmul`'s `outs` semantics: C_out = A*B + C_in, so if C_in is zero, first iteration is correct; subsequent iterations accumulate)

   For the MVP: emit a `linalg.fill` with zero into the output buffer region before the reduction loop. This gets lowered by the standard pipeline.

   > **Alternative**: If the ISA has a "zero" or "fill" instruction, use that. Otherwise, just trust that the buffer is zeroed (document this as a precondition).

2. **Accumulate**: Each reduction loop iteration calls the compute instruction. The instruction reads the current accumulator from the output buffer slot and writes the updated accumulator back to the same slot.

3. **Store**: After the reduction loop completes, store the final accumulated result to HBM.

### Function signature transformation

The source function has tensor arguments and returns:
```mlir
func.func @mm(%A: tensor<256x128xbf16>, %B: tensor<128x64xbf16>) -> tensor<256x64xbf16>
```

After Stage 3, the function operates on buffers:
```mlir
func.func @mm() {
  // act.emit calls referencing act.buffer symbols
  return
}
```

The function arguments are "absorbed" into the HBM layout — their data is assumed to be at the assigned HBM offsets. The return value is in HBM at its assigned offset.

For the MVP: rewrite the function to take no arguments and return nothing. Document the HBM layout alongside the emitted code (via debug logging).

### Cleanup

After emitting all instructions:
1. Erase original source compute ops (`linalg.matmul`, `linalg.generic`, etc.)
2. Erase infrastructure ops that are now dead (`linalg.fill`, `tensor.empty`, `arith.constant` with no remaining uses)
3. DCE pass to clean up any remaining dead ops

---

## Data contiguity constraint

A key constraint for data movement: the load/store instructions typically access **contiguous** memory regions. A subtile of a larger tensor is only contiguous if it spans full innermost dimensions.

For example, loading a 64×64 subtile from a 256×128 matrix:
- Rows are at stride 128 in HBM — the 64-wide rows are NOT contiguous (separated by gaps of 64 elements)
- A contiguous load of 64×64 = 4096 elements would get the wrong data

**When tiles are contiguous**:
- Single invocation (tile = entire tensor): always contiguous
- Tiling only the outermost dims (batch, M for row-major A): row blocks are contiguous
- The innermost dim is not tiled (tile covers the full extent)

**When tiles are NOT contiguous**:
- Tiling an inner dim of a row-major tensor (e.g., tiling K for A[M,K])
- The load instruction would need strided/2D access to gather the subtile

**MVP approach**: The MVP requires tiles to be contiguous for data movement. This restricts which tilings are valid. When emitting load instructions, verify contiguity:

```
operandTileIsContiguous(operand, tileOffsets, tileSizes, tensorStrides):
  // The tile is contiguous iff:
  // For each dim from innermost to outermost:
  //   either tileSize[dim] == tensorShape[dim] (full extent)
  //   or all inner dims are full extent and this is the outermost tiled dim
  for dim in reversed(range(rank)):
    if tileSizes[dim] != tensorShape[dim]:
      // This dim is tiled — all dims below must be full extent
      for inner in range(dim+1, rank):
        if tileSizes[inner] != tensorShape[inner]:
          return false
      return true  // only outermost tiled dim
  return true  // no dim is tiled
```

If a tile is not contiguous: report error in MVP. Future work: support strided loads or insert a packing/unpacking step.

**Special case: instructions operating directly on HBM** (like `@matmul` on `@devmem`): The instruction's addr region already specifies how to access the buffer. The `basis` param is the flat element offset, and the access pattern handles the reshape. In this case, we just need to compute the correct flat offset. Contiguity is handled by the instruction itself (since it accesses a contiguous block and reshapes internally).

---

## Worked Example

### Example 1: batch_matmul with tpu.mlir

**Source**:
```mlir
func.func @test(%a: tensor<4x8x8xf32>, %b: tensor<4x8x8xf32>) -> tensor<4x8x8xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x8x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x8x8xf32>) -> tensor<4x8x8xf32>
  %0 = linalg.batch_matmul ins(%a, %b) outs(%fill) -> tensor<4x8x8xf32>
  return %0 : tensor<4x8x8xf32>
}
```

**ISA**: tpu.mlir with `@matmul` (src(@devmem, @devmem) dst(@devmem), parametric %size)

**Stage 2 result**: structural suffix match, numOuterDims=1, p3=8, tiles=[4,1,1,1]

**Phase 3a**: Coverage resolution — `batch_matmul` → `@matmul`, only valid match.

**Phase 3b**: Memory layout.
- `@devmem` is the single buffer (!act.scalar<f32>, size=8192)
- All tensors in `@devmem` (since @matmul operates on @devmem)
- HBM layout:
  - %a: offset=0, shape=[4,8,8], strides=[64,8,1], size=256
  - %b: offset=256, shape=[4,8,8], strides=[64,8,1], size=256
  - %c: offset=512, shape=[4,8,8], strides=[64,8,1], size=256
- No scratchpad allocation needed (@matmul uses @devmem directly)
- No data movement needed

**Phase 3c**: Code emission.
- Tiling dims:
  - dim 0 (batch): tileFactor=4, parallel → generates loop
  - dim 1 (M): tileFactor=1 → no loop
  - dim 2 (N): tileFactor=1 → no loop
  - dim 3 (K): tileFactor=1, reduction → no loop
- Loop nest: single `scf.for` over batch

**Output**:
```mlir
func.func @test() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  scf.for %batch = %c0 to %c4 step %c1 {
    // A tile: @devmem[batch * 64]
    %a_off = arith.muli %batch, %c64 : index
    // B tile: @devmem[256 + batch * 64]
    %b_off_rel = arith.muli %batch, %c64 : index
    %b_off = arith.addi %c256, %b_off_rel : index
    // C tile: @devmem[512 + batch * 64]
    %c_off_rel = arith.muli %batch, %c64 : index
    %c_off = arith.addi %c512, %c_off_rel : index
    act.emit @matmul addr(%a_off, %b_off, %c_off, 8) compute()
  }
  return
}
```

### Example 2: matmul with qkv.mlir (data movement needed)

**Source**:
```mlir
func.func @mm(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %cst = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<64x64xbf16>
  %fill = linalg.fill ins(%cst : bf16) outs(%empty : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %0 = linalg.matmul ins(%a, %b) outs(%fill) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}
```

**ISA**: qkv.mlir with `@gemm` (src(@d1, @d1) dst(@d2), fixed 64×64)

**Stage 2 result**: same-rank match, all tile factors = 1, no shape params (all constant).

**Phase 3a**: `matmul` → `@gemm`, single valid match.

**Phase 3b**: Memory layout.
- HBM buffer: `@d0` (!act.hbm<16384xbf16>)
  - %a: offset=0, size=4096 bf16 elements
  - %b: offset=4096, size=4096
  - %c: offset=8192, size=4096
- Scratchpad: `@d1` (size=128 vectors of 64xbf16)
  - A slot: offset=0, size=64 vector slots (64 vectors × 64 elements = 64×64 matrix)
  - B slot: offset=64, size=64 vector slots
- Compute output: `@d2` (size=64 vectors of 64xbf16)
  - C slot: offset=0, size=64 vector slots
- Data movement catalog:
  - (d0, d1): `@load_rm` (or `@load_cm`)
  - (d1, d0): `@store_rm`
  - (d1, d2): `@mov` (or if needed, d2→d1 — check ISA)

**Phase 3c**: No tiling loops (all tileFactor=1). Emit load/compute/store sequence.

**Output**:
```mlir
func.func @mm() {
  act.emit @load_rm addr(0, 0, 64) compute()        // A: d0[0] → d1[0], 64 rows
  act.emit @load_rm addr(4096, 64, 64) compute()    // B: d0[4096] → d1[64], 64 rows
  act.emit @gemm addr(0, 64, 0) compute()            // d1[0] × d1[64] → d2[0]
  // Need to move d2[0] → d1[0] for store (store_rm reads from d1)
  // Actually, this depends on ISA having d2→d1 mov... if @mov is d1→d2 only, we have a problem
  // For now, assume we can store directly or have the right mov instruction
  act.emit @store_rm addr(0, 8192, 64) compute()    // d1[0] → d0[8192]
  return
}
```

> **Note**: The `@mov` direction issue (d1→d2 only, but we may need d2→d1) is an ISA limitation that Stage 3 must detect and report. If no identity instruction exists for a required buffer transition, emit a diagnostic.

### Example 3: matmul with tiling and reduction

**Source**: `linalg.matmul ins(tensor<128x128xf32>, tensor<128x128xf32>)` with tpu.mlir `@matmul` (%size param)

**Stage 2**: %size=64 would require M=N=K=64, but source has M=N=K=128. So %size=128 → single invocation? No, the @matmul buffers only read %size*%size contiguous elements...

Actually, for @matmul with `!act.scalar<f32>` buffer of size 8192: %size=128 → needs 128*128=16384 elements per operand, but the buffer only has 8192 slots. So this is infeasible for a single invocation.

If %size=64: tiles [2,2,2], but the subtiles of A, B in @devmem are NOT contiguous (128-wide rows, taking 64-wide subtiles). This hits the contiguity constraint.

This example shows the limitation of tpu.mlir's flat access pattern for tiled matmul. It works only when the tile spans the full tensor or when batch tiling preserves contiguity.

---

## Implementation Plan

### Files to create/modify

| File | Change |
|------|--------|
| `mlir/include/act/Support/CodeEmission.h` | New: declare `runCodeEmission`, data structures |
| `mlir/lib/act/Support/CodeEmission.cpp` | New: implement all three phases |
| `mlir/lib/act/Conversion/ConvertCanonicalFormToAct.cpp` | Wire Stage 3 after Stage 2 |
| `mlir/lib/act/Support/CMakeLists.txt` | Add CodeEmission.cpp |

### Iteration 1: Direct-buffer instructions, single compute op

**Scope**: Handle instructions where all operands are in the same buffer (no data movement). Handle batch tiling (outer parallel loops). Single compute op per function.

Steps:

#### Step 1: Coverage resolution
- For each source compute op, find valid TiledMatchCandidates, pick best
- Build `InstructionPlan`

#### Step 2: HBM layout assignment
- Walk function args, assign sequential offsets in the single buffer
- Compute row-major strides for each tensor

#### Step 3: Loop nest emission
- Build list of tiling dims with tileFactor > 1
- Separate into parallel (outer) and reduction (inner)
- Emit nested `scf.for` ops

#### Step 4: Compute addr params
- For offset params: compute flat HBM offset from loop IVs + tensor strides + indexing maps
- For shape params: use solved values from tiling scheme
- Map to correct positions in the addr block arg list

#### Step 5: Emit act.emit
- Create `act.emit` op with computed addr params
- Erase original source ops + dead infrastructure

#### Step 6: Function signature
- Remove tensor args and returns
- The function body is now purely buffer-level

**Test**: `batch_mm.mlir` (4×8×8) with tpu.mlir → batch loop + `act.emit @matmul`

### Iteration 2: Data movement

**Scope**: Handle instructions with different src/dst buffers. Insert load/store around compute.

Steps:

#### Step 7: Build data movement catalog
- Identify identity instructions from ISA
- Build (srcBuffer, dstBuffer) → DefineOp map

#### Step 8: Emit load/store instructions
- Before compute: for each src operand, if data is not in src buffer, emit load
- After compute (outside reduction loops): for each dst operand, emit store
- Compute addr params for load/store based on HBM tile offset and scratchpad slot

#### Step 9: Handle buffer transitions
- If compute dst buffer ≠ store src buffer, insert mov instruction
- Report error if no identity instruction exists for required transition

**Test**: `mm.mlir` (64×64 bf16) with qkv.mlir → load_rm + gemm + store_rm sequence

### Iteration 3: Multi-op functions

**Scope**: Handle functions with multiple compute ops and intermediate results.

Steps:

#### Step 10: Topological ordering
- Order plan entries by data dependencies (SSA use-def chains)
- Handle linalg.fill as accumulator initialization (tied to its consumer)

#### Step 11: Intermediate HBM allocation
- Allocate HBM space for intermediate tensor values (between compute ops)
- Track which HBM region each intermediate occupies

#### Step 12: Chain emission
- Emit instructions for each compute op in topological order
- Each op's load/store instructions reference the HBM regions of its operands
- Intermediate results: stored to HBM after one op, loaded from HBM before the next

**Test**: `attention.mlir` → sequence of load/compute/store for each op in the attention graph

---

## Edge cases and constraints

### Padding
If `sourceBound % nativeValue != 0` (flagged by `needsPadding` in TilingScheme), the last tile is smaller than the native size. For the MVP: reject such cases (require divisible tiling). Future: emit padding logic or use ISA-specific partial-tile instructions.

### Fill absorption
`linalg.fill` is not a compute instruction — it initializes an accumulator. In Stage 3:
- If the fill's output feeds directly into a compute op's `outs` operand: the fill is absorbed into the accumulator initialization (zero-init the scratchpad slot or HBM region before the reduction loop)
- The fill op itself is erased after emission

### Dynamic shapes
The MVP assumes all shapes are statically known. Dynamic shapes would require runtime computation of tiling factors and HBM offsets. Deferred.

### Contiguity validation
Before emitting a load instruction, validate that the tile being loaded is contiguous in HBM. If not, either:
1. For instructions operating directly on their buffer (no separate load): compute the correct flat offset and trust the instruction's access pattern
2. For separate load instructions: report error (future: emit packing/strided-load)
