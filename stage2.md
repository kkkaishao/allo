# Stage 2: Tiling and Address Parameter Generation — Detailed Design

## Status

**Iteration 1 (same-rank tiling): COMPLETE** (2026-04-13)
**Iteration 2 (rank mismatch handling): COMPLETE** (2026-04-13)

Implemented files:
- `mlir/include/act/Support/SymbolicExpr.h` — `SymExpr` struct, `buildSymExpr`, `generateShapeExpr`, `SymShape`
- `mlir/lib/act/Support/SymbolicExpr.cpp` — SSA tracing, per-op shape generation (StridedOp, ExpandShapeOp, CollapseShapeOp, TransposeOp)
- `mlir/include/act/Support/SemanticMatching.h` — `MatchCandidate` (with `numOuterDims`), `runStructuralMatching` declaration
- `mlir/lib/act/Support/SemanticMatching.cpp` — structural suffix matching (body equiv, iterator suffix, indexing map compat)
- `mlir/include/act/Support/TilingAnalysis.h` — `TilingScheme`, `TiledMatchCandidate` (with `numOuterDims`), `AddrParamKind`, `runTilingAnalysis`
- `mlir/lib/act/Support/TilingAnalysis.cpp` — symbolic iteration domain extraction, constraint solving, tiling factor computation, param classification, rank mismatch tiling
- `mlir/lib/act/Conversion/ConvertCanonicalFormToAct.cpp` — wiring: structural matching between Stage 1 and Stage 2, debug logging

Design deviations from this doc:
- `generateShapeExpr` is a **free function** dispatching via `isa<>` checks (in `SymbolicExpr.cpp`), not an interface method on `BufferAccessOpInterface`. This avoids header dependency between IR and Support; can be promoted later.
- Source iteration domain uses `LinalgOp::getStaticLoopRanges()` (from `IndexingMapOpInterface`) instead of `TilingInterface::getIterationDomain()`. Simpler and sufficient for static shapes.
- Phase 2b is lightweight: classifies params as Shape/Offset/Mixed only. Offset expression computation deferred to Stage 4 (code emission).

## Goal
For each match candidate from Stage 1, determine valid tiling factors and address parameters that make the source tensor shapes fit the instruction's constraints. Output tiled match candidates annotated with loop structure, tile sizes, and per-tile addr param expressions.

## Input / Output
- **Input**: `SmallVector<MatchCandidate>` from Stage 1 (each is a `(sourceOp, DefineOp)` pair), plus buffer/ISA declarations in the module.
- **Output**: `SmallVector<TiledMatchCandidate>` — each annotated with tiling factors, loop nest structure, solved shape params, and symbolic offset expressions for addr params.

## Architecture: Two Phases

### Phase 2a: Symbolic shape analysis + tiling
Determine the instruction's native computation shape via symbolic expressions, compare against the source shape, and compute tiling factors.

### Phase 2b: Address parameter binding
For each tiled invocation, determine concrete addr param expressions as functions of tiling loop induction variables.

---

## Symbolic Expression Framework

### Data structure: `SymExpr`

A tree representing integer expressions over addr parameters. Placed in `Support/SymbolicExpr.{h,cpp}`.

```cpp
struct SymExpr {
  enum Kind { Constant, Param, Add, Mul };
  Kind kind;
  int64_t value;                    // Constant
  unsigned paramIdx;                // Param: index into addr block args
  std::shared_ptr<SymExpr> lhs, rhs; // Add, Mul

  static SymExpr constant(int64_t v);
  static SymExpr param(unsigned idx);
  static SymExpr add(SymExpr a, SymExpr b);
  static SymExpr mul(SymExpr a, SymExpr b);

  // Evaluate with concrete param values
  int64_t evaluate(ArrayRef<int64_t> paramValues) const;

  // Collect param indices that appear in this expression
  void collectParams(DenseSet<unsigned> &out) const;

  // Constant-fold where possible
  SymExpr simplify() const;

  // For export to external solvers
  std::string toString(ArrayRef<StringRef> paramNames = {}) const;
  
  // Query
  bool isConstant() const;
  bool isParam() const;
};
```

For the MVP, `Add` and `Mul` cover all addr regions in tpu.mlir and qkv.mlir (`arith.addi`, `arith.muli`). Future extensions: `Div`, `Mod`, `Min`, `Max`.

### `SymShape`: symbolic tensor shape

```cpp
using SymShape = SmallVector<SymExpr>; // one SymExpr per tensor dimension
```

### Building `SymExpr` from SSA values

```cpp
FailureOr<SymExpr> buildSymExpr(Value v);
```

Walks backward through the addr region's SSA:
- `BlockArgument` at position `i` → `Param(i)`
- `arith.constant c` → `Constant(c)`
- `arith.muli %a, %b` → `Mul(build(%a), build(%b))`
- `arith.addi %a, %b` → `Add(build(%a), build(%b))`
- Anything else → `failure()` (unsupported; clean error)

### `generateShapeExpr` — free function

> **Implementation note**: Originally designed as an interface method on `BufferAccessOpInterface`, but implemented as a free function in `Support/SymbolicExpr.{h,cpp}` to avoid header dependency between IR and Support. Can be promoted to an interface method later if needed.

```cpp
FailureOr<SymShape> generateShapeExpr(Operation *accessOp, BufferTypeInterface bufferType);
```

Dispatches via `isa<>` checks. Each access op type has its own implementation.

#### `act.strided` — base access

The strided op produces a 1D access. The buffer type contributes trailing dimensions:
- `!act.scalar<f32>` (rank 0): shape = `[C]`
- `!act.vector<64xbf16>` (rank 1, element shape [64]): shape = `[C, 64]`
- `!act.tile<4x4xf32>` (rank 2, element shape [4, 4]): shape = `[C, 4, 4]`
- `!act.hbm<16384xbf16>` (rank 1, element shape [16384]): depends on buffer size — if size=1, the strided access addresses sub-elements, shape = `[C]`

Where `C` = `buildSymExpr(counts[0])`.

General rule: `shape = [counts...] ++ bufferType.getShape()`, except for HBM with size=1 where the buffer represents a flat address space and the element shape describes the total capacity, not per-slot shape.

Note: multi-dimensional strided access (multiple counts) is possible but rare. For the MVP we handle 1D strided (single count dimension), which covers all current ISAs.

#### `act.expand_shape` — refine dimensionality

```
Input shape [C, ...] → Output shape [D0, D1, ...]
```

Replaces dimensions according to reassociation indices. The output shape dimensions are explicit in the op's `output_shape` attribute.

```cpp
FailureOr<SymShape> ExpandShapeOp::generateShapeExpr(BufferTypeInterface bufTy) {
  // Delegate to source for input shape (unused directly — output shape is explicit)
  SymShape result;
  for (auto dim : getMixedOutputShape())
    result.push_back(buildSymExpr(dim));  // static → Constant, dynamic → trace SSA
  return result;
}
```

Wait — expand_shape only affects the reassociated dimensions. Non-reassociated dimensions pass through from the source shape. So the correct implementation must:
1. Get source shape from `getSource().getDefiningOp<BufferAccessOpInterface>()->generateShapeExpr(bufTy)`
2. For each reassociation group, replace the source dim(s) with the output dims from `output_shape`
3. Non-reassociated dims pass through unchanged

For the common case `[[0, 1]] output_shape [D0, D1]` on a 1D source `[C]`:
- Source shape: `[C]`
- Reassociation: dim 0 → [dim 0, dim 1]
- Result: `[D0, D1]` where `D0`, `D1` come from `output_shape`

#### `act.collapse_shape` — reduce dimensionality

```
Input shape [D0, D1, ...] → Output shape [D0*D1, ...]
```

Merges dimensions according to reassociation. The output dimensions are products of the merged input dimensions.

```cpp
FailureOr<SymShape> CollapseShapeOp::generateShapeExpr(BufferTypeInterface bufTy) {
  auto sourceShape = ...; // get from source
  SymShape result;
  for (auto &group : getReassociationIndices()) {
    SymExpr merged = sourceShape[group[0]];
    for (unsigned i = 1; i < group.size(); ++i)
      merged = SymExpr::mul(merged, sourceShape[group[i]]);
    result.push_back(merged);
  }
  // Append non-reassociated trailing dims
  return result;
}
```

#### `act.transpose` — permute dimensions

```
Input shape [D0, D1] → Output shape [D_perm[0], D_perm[1]]
```

```cpp
FailureOr<SymShape> TransposeOp::generateShapeExpr(BufferTypeInterface bufTy) {
  auto sourceShape = ...; // get from source
  auto perm = getPermutation();
  SymShape result(perm.size());
  for (unsigned i = 0; i < perm.size(); ++i)
    result[i] = sourceShape[perm[i]];
  return result;
}
```

### Top-level: `computeSymbolicShape`

For each yield operand in the addr region, walk the access chain and call `generateShapeExpr`:

```cpp
FailureOr<SymShape> computeSymbolicShape(Value yieldOperand,
                                          BufferTypeInterface bufferType) {
  auto accessOp = yieldOperand.getDefiningOp<BufferAccessOpInterface>();
  return accessOp.generateShapeExpr(bufferType);
}
```

Each op's `generateShapeExpr` recursively calls its source's `generateShapeExpr`, so the chain is walked automatically.

---

## Phase 2a: Tiling Analysis

### Step 1: Extract instruction's symbolic shapes

For each matched instruction, walk all yield operands:

```
for i in 0..numOperands:
  yieldOperand = addrBlock.getTerminator()->getOperand(i)
  bufferType = getBufferTypeForOperand(defineOp, i)  // from src/dst declarations
  symShapes[i] = computeSymbolicShape(yieldOperand, bufferType)
```

### Step 2: Map symbolic shapes to iteration domain

The instruction's compute region contains a linalg op with indexing maps. These maps relate operand dimensions to iteration dimensions:

```
For @gemm:
  compute has linalg.matmul with maps: A→(M,K), B→(K,N), C→(M,N)
  symShapes: A=[64,64], B=[64,64], C=[64,64]
  → iteration bounds: M=64, N=64, K=64
```

For each iteration dimension `d`, collect all constraints from all operands:
```
constraints[d] = {symExpr | operand i, operand dim j maps to iteration dim d}
```

All constraints for the same iteration dim must be consistent. If they involve different SymExprs (e.g., one says M=Param(3), another says M=Constant(64)), that's an error in the ISA definition.

The result is a **symbolic iteration domain**: `SymExpr` bounds for each iteration dimension of the instruction.

### Step 3: Compare against source iteration domain

The source linalg op's iteration domain gives concrete bounds (via `TilingInterface::getIterationDomain()`):

```
Source linalg.matmul: M=256, N=64, K=128
Instruction @gemm:    M=64,  N=64, K=64
```

For each iteration dimension:
- If `symExpr` is `Constant(c)`: native bound = `c`. If `source_bound > c`, tiling factor = `ceil(source_bound / c)`.
- If `symExpr` is `Param(i)`: the param can be set to `source_bound`. No tiling needed for this dim (the instruction adapts). Record constraint: `params[i] = source_bound`.
- If `symExpr` is a compound expression: solve for the free params. If multiple dims constrain the same param to different values, the param takes the GCD (or the minimum) and the remaining extent is tiled.

### Step 4: Solve constraint system

Collect all constraints from Step 3:
```
params[i] = value_1  (from dim d1)
params[i] = value_2  (from dim d2)
...
```

For the MVP solver:
1. **Single-param, single-value**: direct assignment.
2. **Single-param, multiple values**: if all values equal → consistent. If not → the param takes the minimum (or GCD if divisibility is needed), and remaining extent is tiled.
3. **Multi-param**: if each param is independently constrained → solve independently. If params are coupled (e.g., `Param(0) * Param(1) = 128`) → defer to external solver or bounded enumeration.

### Step 5: Compute tiling factors

For each iteration dimension `d`:
```
native_bound = symIterBound[d].evaluate(solvedParams)
source_bound = sourceIterDomain[d]
if source_bound > native_bound:
  tile_factor = ceil(source_bound / native_bound)
  if source_bound % native_bound != 0:
    flag for padding analysis
else:
  tile_factor = 1  // fits in one invocation
```

### Output of Phase 2a

```cpp
struct TilingScheme {
  // Solved shape params: paramIdx → concrete value
  DenseMap<unsigned, int64_t> solvedParams;

  // Per iteration dimension: source bound, native bound, tile factor
  struct DimTiling {
    int64_t sourceBound;
    SymExpr nativeBound;   // symbolic (may reference solved params)
    int64_t tileFactor;
    bool needsPadding;
  };
  SmallVector<DimTiling> dims;

  // Iterator types (parallel/reduction) from the source op
  SmallVector<utils::IteratorType> iteratorTypes;
};
```

---

## Phase 2b: Address Parameter Binding

Given a tiling scheme, determine addr param expressions for each tiled invocation.

### Classifying addr params

Each addr parameter falls into one of:
1. **Shape params**: appear in `counts` or `output_shape` of access ops. Their values are determined by Phase 2a's constraint solver. Constant across all tile invocations.
2. **Offset params**: appear in `basis` of `act.strided`. Their values depend on which tile is being processed. Vary with loop induction variables.

Classification is done by examining where each param appears in the symbolic expressions:
```cpp
for each yield operand:
  walk the access chain:
    for act.strided: basis params → offset, counts params → shape
    for act.expand_shape: output_shape params → shape
    for act.collapse_shape: (no new params)
    for act.transpose: (no new params)
```

### Computing offset expressions

For offset params, we need to express them as functions of the tiling loop IVs. The key relationship:

```
basis_param = base_address + f(loop_IVs, tile_sizes, tensor_strides)
```

For a concrete operand with indexing map `(M, N, K) → (M, K)`:
- When tiling M with step `tile_M` and K with step `tile_K`:
- The tile at `(iv_m, iv_k)` accesses rows `[iv_m, iv_m+tile_M)` and cols `[iv_k, iv_k+tile_K)`
- In a flat buffer (scalar), the offset = `iv_m * stride_row + iv_k * stride_col`
- In a vector buffer with element size `E`, the offset = `iv_m * stride_row_in_slots + iv_k * stride_col_in_slots`

The stride information comes from the source tensor's layout. For the common case of contiguous row-major storage in a buffer:
- stride_row = number of buffer slots per row = tensor_dim_1 / buffer_element_size (for vector buffers)
- stride_col = 1 (contiguous within a row)

This part is tightly coupled with data layout and buffer allocation, which may be a separate concern. For the MVP:
- Assume source tensors are already in the right buffers at known offsets
- Offset params are simple: `base + iv * tile_size` for contiguous layouts
- Non-contiguous layouts (sub-tiles of larger tensors) require data movement instructions (load/store) to pack tiles into contiguous scratchpad slots first

### Output of Phase 2b

```cpp
struct AddrBinding {
  // For each addr param: either a solved constant or an expression over loop IVs
  SmallVector<AddrParamExpr> params;
};

struct AddrParamExpr {
  enum Kind { Constant, LoopExpr };
  Kind kind;
  int64_t constantValue;           // for solved shape params
  // For offset params: symbolic expression over loop IVs
  // (details TBD — depends on buffer allocation strategy)
};
```

---

## Combined Output: TiledMatchCandidate

```cpp
struct TiledMatchCandidate {
  MatchCandidate base;          // from Stage 1
  TilingScheme tiling;          // from Phase 2a
  AddrBinding addrBinding;      // from Phase 2b

  // Convenience: does this need tiling at all?
  bool needsTiling() const {
    return llvm::any_of(tiling.dims, [](auto &d) { return d.tileFactor > 1; });
  }
};
```

---

## Rank Mismatch Handling (Iteration 2)

Not implemented in iteration 1, but the design supports it.

### Problem

Source: `linalg.batch_matmul` with iteration domain `(batch, M, N, K)` — types `(parallel, parallel, parallel, reduction)`.
Instruction: `@gemm` with iteration domain `(M, N, K)` — types `(parallel, parallel, reduction)`.

Stage 1 won't match these by name (different ops). But structurally, the inner computation is identical.

### Approach: Structural iteration domain matching

Instead of matching by op name alone, compare the **iteration domain structure**:

1. Extract the instruction's iteration types from its compute region's linalg op: `(parallel, parallel, reduction)` for `@gemm`.
2. For an unmatched source op, extract its iteration types: `(parallel, parallel, parallel, reduction)` for `batch_matmul`.
3. Try to find a **suffix match**: the instruction's iteration types match a contiguous suffix of the source's iteration types.
4. If found, the unmatched prefix dims are all parallel → they become outer loops.

More precisely, for a source with iteration types `[t_0, t_1, ..., t_{n-1}]` and instruction with types `[u_0, ..., u_{m-1}]` where `m < n`:
- Check that `t_{n-m+i} == u_i` for all `i in 0..m-1` (suffix match on types)
- Check that the indexing maps are compatible: the instruction's indexing maps, when padded with leading identity dims, match the source's indexing maps
- Check that the extra dims `t_0, ..., t_{n-m-1}` are all `parallel` (reduction dims can't be trivially peeled)

If all checks pass:
- The extra dims become `scf.for` loops with full extent (tile_factor = source_bound for those dims)
- The inner dims are handled by the standard Phase 2a tiling

### Indexing map compatibility check

For `batch_matmul`:
- Indexing maps: `A→(b,M,K)`, `B→(b,K,N)`, `C→(b,M,N)`

For `@gemm` (matmul):
- Indexing maps: `A→(M,K)`, `B→(K,N)`, `C→(M,N)`

The batch dim `b` appears as a leading identity dim in all operands. Stripping it from each map produces the instruction's maps. This is the compatibility check: the extra dims must be leading dims that appear identically in all operand indexing maps (i.e., the batch structure).

### Integration with TilingInterface

The outer (batch) loops are generated using linalg's `TilingInterface`:
1. Set tile sizes for outer dims = 1 (peel one batch element at a time), inner dims = instruction's native bounds
2. Call `getTiledImplementation()` → produces a tiled linalg op with extracted slices for each operand
3. The inner tiled op is now rank-reduced and matches the instruction exactly

This reuses MLIR's tiling infrastructure rather than reimplementing slice extraction.

### Why not a pre-pass?

A pre-pass that converts `batch_matmul` → `scf.for + matmul` before Stage 1 is simpler but inflexible:
- Destroys the chance to match an ISA that natively supports `batch_matmul`
- Hard-codes which ops to decompose
- Doesn't generalize to other rank mismatches (e.g., `conv_2d` matched by a `matmul` instruction via im2col)

The structural approach in Stage 2 is unified: any rank mismatch with compatible iteration structure is handled the same way.

---

## Implementation Plan

### Iteration 1: Same-rank tiling (MVP) ✓ COMPLETE

**Scope**: Handle the case where source op rank matches instruction rank. All tpu.mlir and qkv.mlir instructions fall into this category.

#### Step 1: SymExpr data structure ✓
- `SymExpr` with `Constant`, `Param`, `Add`, `Mul` + constant folding (add/mul identity, constant propagation)
- `buildSymExpr(Value)` traces addr region SSA backward
- `buildSymExpr(OpFoldResult)` handles mixed static/dynamic values
- `evaluate()`, `collectParams()`, `toString()`, `getConstantValue()`, `getParamIdx()`

#### Step 2: `generateShapeExpr` free function ✓
- Implemented as free function dispatching via `isa<>` (not interface method — avoids IR↔Support header dependency)
- `StridedOp`: `[counts...] ++ bufferType.getShape()` for non-HBM; rank reduction drops leading Constant(1) when trailing element dims exist
- `ExpandShapeOp`: recursive source shape, replace per reassociation with `buildSymExpr(outputShape[idx])`
- `CollapseShapeOp`: recursive source shape, merge per reassociation with `SymExpr::mul`
- `TransposeOp`: recursive source shape, permute via `result[i] = sourceShape[perm[i]]`
- `TiledOp`: deferred (no current ISA uses it in addr regions)

#### Step 3: Phase 2a — tiling analysis ✓
- `extractSymbolicShapes(DefineOp, ModuleOp)`: walks addr yield operands with buffer type lookup
- `extractSymbolicIterationDomain(DefineOp, symShapes)`: maps symbolic shapes to iteration dims via linalg indexing maps (AffineDimExpr projections only)
- `getSourceIterationDomain(LinalgOp)`: uses `getStaticLoopRanges()` (requires all static dims for MVP)
- `computeTilingScheme()`: Pass 1 solves Param constraints (direct assignment + consistency check), Pass 2 evaluates bounds and computes `ceil(source/native)` tile factors with padding detection
- Correctly identifies infeasible matches (e.g., @matmul with M≠N≠K)

#### Step 4: Phase 2b — address parameter classification ✓
- `classifyAddrParams(DefineOp)`: walks addr block ops, marks basis params as Offset, counts/output_shape params as Shape, both as Mixed
- Offset expression computation deferred to Stage 4

#### Step 5: Integration into the pass ✓
- Wired into `ConvertCanonicalFormToAct` pass after Stage 1 matching
- LLVM_DEBUG output: validity, solved params, tiling factors, padding flags, param kinds
- Tested end-to-end with tpu.mlir and qkv.mlir ISAs against source linalg ops

### Iteration 2: Rank mismatch handling ✓ COMPLETE

#### Step 6: Structural suffix matching ✓
- `runStructuralMatching(module, unmatchedOps, results)` in `SemanticMatching.cpp`: for each unmatched linalg op × each DefineOp, checks suffix match + body equiv + indexing map compat
- `checkIteratorTypeSuffix(sourceTypes, instrTypes)`: verifies instruction iter types are a suffix of source's with all-parallel prefix; returns offset (numOuterDims)
- `checkBodyEquivalence(sourceOp, instrOp)`: manual lockstep body comparison with value ID mapping (block args by index, SSA results by definition order). `OperationEquivalence::isRegionEquivalentTo` was tried first but returned false on identical bodies — root cause unclear, replaced with manual comparison
- `checkIndexingMapCompatibility(sourceOp, instrOp, offset)`: strips results referencing batch dims (d0..d_{offset-1}) from source maps, reindexes remaining dims via `AffineMap::replaceDimsAndSymbols`, compares against instruction maps
- `findComputeLinalgOp(DefineOp)`: extracts single linalg op from compute region (factored out)
- `numOuterDims` field added to `MatchCandidate` (>0 for structural suffix matches)

#### Step 7: Rank mismatch tiling ✓
- TilingAnalysis handles `numOuterDims > 0`: extracts inner source bounds and iter types (suffix), calls `computeTilingScheme` on inner portion, prepends outer dims with nativeValue=1 and tileFactor=sourceBound
- `computeTilingScheme` refactored from `(linalg::LinalgOp sourceOp, ...)` to `(ArrayRef<utils::IteratorType> iterTypes, ...)` to support passing inner (suffix) iterator types
- `numOuterDims` propagated to `TiledMatchCandidate` for downstream use
- Note: TilingInterface integration (generating actual `scf.for` loops) deferred to Stage 4 (code emission). Iteration 2 computes the tiling scheme; loop generation is a code emission concern.

Design deviations from this doc:
- Body equivalence uses manual lockstep comparison, not `OperationEquivalence::isRegionEquivalentTo` (see Step 6 note)
- Outer loops are not generated yet — only tiling factors computed. Actual loop generation is a Stage 4 concern.

Verified end-to-end:
- `batch_matmul<4x8x8xf32>` → `@matmul` (tpu.mlir): structural match, numOuterDims=1, p3=8, tiles=[4,1,1,1]
- `batch_matmul<1x64x64xf32>` → `@matmul` (tpu.mlir): structural match, numOuterDims=1, p3=64, single-invocation
- `matmul<10x20x30xf32>` → `@matmul` (tpu.mlir): same-rank path unchanged, correctly INFEASIBLE (M≠N≠K)

### Iteration 3: Advanced solving

#### Step 8: Export to external solver
- Serialize constraint system (as SymExpr trees) to a format consumable by Z3 / OR-Tools / custom solver
- Handle coupled multi-param constraints
- Handle non-linear constraints (e.g., `Param(0) * Param(1) = N`)

---

## Worked Examples

### Example 1: Fixed-shape instruction (no solving needed)

```
Source: linalg.matmul ins(tensor<256x128xbf16>, tensor<128x64xbf16>) outs(tensor<256x64xbf16>)
Instruction: @gemm from qkv.mlir
```

Symbolic shapes from @gemm addr region:
- A: `act.strided basis(%0) counts(64)` on `!act.vector<64xbf16>` → `[Const(64), Const(64)]`
- B: `act.strided basis(%1) counts(64)` on `!act.vector<64xbf16>` → `[Const(64), Const(64)]`
- C: `act.strided basis(%2) counts(64)` on `!act.vector<64xbf16>` → `[Const(64), Const(64)]`

Compute linalg.matmul indexing maps: A→(M,K), B→(K,N), C→(M,N)
→ Symbolic iteration domain: M=64, N=64, K=64 (all constant)

Source iteration domain: M=256, N=64, K=128

Tiling:
- M: 256/64 = 4, tile_factor=4
- N: 64/64 = 1, tile_factor=1
- K: 128/64 = 2, tile_factor=2

Addr params: all offset params (`%addr_a`, `%addr_b`, `%addr_c`)
- `%addr_a = iv_m * 64/64 + iv_k * ...` (depends on buffer layout of A in @d1)
- Shape params: none (all fixed)

### Example 2: Dynamic-shape instruction (single param)

```
Source: linalg.matmul ins(tensor<8x8xf32>, tensor<8x8xf32>) outs(tensor<8x8xf32>)
Instruction: @matmul from tpu.mlir
```

Symbolic shapes from @matmul addr region:
- A: `strided(counts=%size*%size) → expand_shape [%size, %size]` → `[Param(3), Param(3)]`
- B: same → `[Param(3), Param(3)]`
- C: same → `[Param(3), Param(3)]`

Compute linalg.matmul indexing maps: A→(M,K), B→(K,N), C→(M,N)
→ Symbolic iteration domain: M=Param(3), N=Param(3), K=Param(3)

Source iteration domain: M=8, N=8, K=8

Constraints: Param(3)=8 from M, Param(3)=8 from N, Param(3)=8 from K → consistent.

Solved: `%size = 8`. Tile factors: all 1 (single invocation).

Addr params: `%addr_a=base_a, %addr_b=base_b, %addr_c=base_c, %size=8`

### Example 3: Dynamic-shape, inconsistent constraints

```
Source: linalg.matmul ins(tensor<16x8xf32>, tensor<8x12xf32>) outs(tensor<16x12xf32>)
Instruction: @matmul from tpu.mlir
```

Symbolic iteration domain: M=Param(3), N=Param(3), K=Param(3)

Source: M=16, N=12, K=8

Constraints: Param(3)=16, Param(3)=12, Param(3)=8 → inconsistent (three different values).

This instruction requires square matrices (M=N=K). It **cannot** tile this source at all — the fundamental constraint is that M, N, K must be equal. Report as infeasible match.

(If the ISA had a matmul instruction with separate M, N, K params, it would work. The symbolic framework correctly identifies this limitation.)

### Example 4: Rank mismatch (iteration 2)

```
Source: linalg.batch_matmul ins(tensor<4x64x128xbf16>, ...) outs(tensor<4x64x64xbf16>)
Instruction: @gemm from qkv.mlir (symbolic iteration domain: M=64, N=64, K=64)
```

Source iteration types: `(parallel, parallel, parallel, reduction)` — dims (batch=4, M=64, N=64, K=128)
Instruction iteration types: `(parallel, parallel, reduction)` — 3 dims

Suffix match: source dims 1-3 types `(parallel, parallel, reduction)` == instruction types ✓
Extra dim: source dim 0 is `parallel` ✓

Indexing maps compatibility: batch dim appears as leading identity in all source operand maps ✓

Result: outer loop over batch (4 iterations), inner tiling as Example 1 with K: 128/64=2.

### Example 5: Identity instruction (data movement)

```
Source: (not matched by Stage 1 — identity instructions are used for data movement, not computation)
Instruction: @load_rm from qkv.mlir
```

Symbolic shapes:
- Source (from @d0, HBM): `strided(counts=%arg2*64) → expand_shape [%arg2, 64]` → `[Param(2), Const(64)]`
- Dest (to @d1, vector): `strided(counts=%arg2)` on `!act.vector<64xbf16>` → `[Param(2), Const(64)]`

This instruction isn't used in compute matching — it's used by Stage 4 (code emission) when generating data movement. But the symbolic framework still works: given a tile of size `[N, 64]` to load, solve `Param(2) = N`, and `Param(0) = HBM_offset`, `Param(1) = scratchpad_slot`.
