# Stage 1: Semantic Matching — Detailed Design [Version 2]

## Goal
For each compute op (or connected subgraph) in the source computation graph, determine which `act.define` instruction(s) can implement it. Output a list of match candidates mapping source ops to instruction names with SSA bindings. No tiling, no address params, no code emission — purely semantic.

## Pass invocation
```
act-opt computation.mlir -convert-canonical-form-to-act="isa-path=/path/to/isa.mlir"
```
`isa-path` is an optional argument pointing to the MLIR file containing `act.define` instruction definitions. If not provided, it assumes `computation.mlir` also contains the `act.define`s. This allows flexibility in how the user organizes their MLIR files.

The pass is a "meta" pass that orchestrates all the stages of the whole instruction selection pipeline, and this file describes the first stage — semantic matching. The pass will later invoke other stages. The implementation of current stage should be modular and self-contained, and placed under `Support/` folder.

## Pre-pass pipeline
Run on the source module before matching:
```
specialize-generic-ops → canonicalize → cse
```
- `specialize-generic-ops`: converts `linalg.generic` back to named ops (`linalg.matmul`, `linalg.add`, etc.) where possible. Uses upstream `populateLinalgGenericOpsSpecializationPatterns` or exising pass `linalg-specialize-generic-ops`. This is the key enabler — after specialization, most source ops are named, and matching is name comparison.
- `canonicalize`: normalizes patterns like `cmpf+select` → `maximumf`, so that source and instruction bodies use the same form.
- `cse`: eliminates redundant ops to simplify the graph.

## Fingerprint extraction from `act.define`

### Step 0: Validate compute region (boundary layout detection)

After the compute/access separation (see "Design Decision" in thoughts.md), compute regions must contain only computation — all layout transforms belong in the addr region. The boundary stripping logic is retained as a **validator** that detects ISA authoring errors.

The validator classifies layout ops in the compute region as:
- **Pre-transforms**: layout ops whose operands are *only* block arguments, other pre-transforms, or constants. Traced forward from block args via fixed-point iteration.
- **Post-transforms**: layout ops whose results *only* feed into `act.yield` or other post-transforms. Traced backward from yield via fixed-point iteration.

Layout ops are: `tensor.expand_shape`, `tensor.collapse_shape`, `tensor.extract_slice`, `tensor.insert_slice`, `linalg.transpose`.

If any pre/post-transforms are found, the validator emits a warning:
```
warning: compute region of @matmul contains boundary layout op 'tensor.expand_shape'
         that should be in the addr region (pre-transform)
```

This catches the common mistake of putting buffer-to-compute shape adaptation in the compute region instead of the addr region. Interior layout ops (e.g., transpose feeding into matmul in a fused instruction) are *not* flagged — they are legitimate computation.

Example — **correct** `@matmul` (after separation):
```mlir
compute(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) {
    %0 = linalg.matmul ins(%a, %b) outs(%c) -> tensor<4x4xf32>
    act.yield %0
}
```
No layout ops → no warnings. Core: `linalg.matmul`. Fingerprint: `linalg.matmul`.

Example — **incorrect** `@matmul` (pre-separation style, now flagged):
```mlir
compute(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<16xf32>, %size: index) {
    %0 = tensor.expand_shape %a ...   // WARNING: pre-transform
    %1 = tensor.expand_shape %b ...   // WARNING: pre-transform
    %2 = tensor.expand_shape %c ...   // WARNING: pre-transform
    %3 = linalg.matmul ...            // core compute
    %4 = tensor.collapse_shape %3 ... // WARNING: post-transform
    act.yield %4
}
```
3 pre-transforms, 1 post-transform → 4 warnings emitted.

Example — **correct** fused `@transpose_matmul`:
```mlir
compute(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) {
    %empty = tensor.empty() : tensor<4x4xf32>
    %0 = linalg.transpose ins(%a) outs(%empty) permutation = [1, 0]
    %1 = linalg.matmul ins(%0, %b) outs(%c) -> tensor<4x4xf32>
    act.yield %1
}
```
No boundary layout ops (transpose feeds into matmul, not yield — it's interior). Core: `linalg.transpose → linalg.matmul`. Fingerprint: multi-op (not yet supported, falls back to Identity).

### Step 1: Build the semantic fingerprint
After validation, the core ops are collected by filtering out yield, constants, and `tensor.empty` (allocation infrastructure, not compute). All remaining ops form the fingerprint.

**Single-op fingerprint** (most common):
- If the core is a single named linalg op (e.g., `linalg.matmul`): fingerprint = op name.
- If the core is a single `linalg.generic`: fingerprint = (indexing maps, iterator types, body hash).

**Multi-op fingerprint** (for fused instructions):
- A small DAG of single-op fingerprints connected by data flow edges.
- Matched by subgraph isomorphism against the source graph.

**Identity fingerprint** (for load/store instructions):
- Compute region is just `act.yield %input` — no compute ops.
- These are data movement instructions, not matched against source compute ops. They're used by later stages for data movement generation.

### Step 3: Normalize for comparison
For both source ops and instruction fingerprints:
1. If named op: use the op name directly as the key.
2. If `linalg.generic`: compute a structural hash:
   - Hash = f(numInputs, numOutputs, indexingMaps, iteratorTypes, bodyHash)
   - `indexingMaps`: compared structurally (same affine expressions), independent of tensor shapes.
   - `bodyHash`: structural hash of the scalar body — each op hashed as `(opName, operandIndices)` in topological order. Constants are hashed by value.
3. Two fingerprints match if their hashes match AND a detailed structural comparison confirms equivalence.

## Matching algorithm

### Build instruction catalog
```
catalog: Map<fingerprint_key, List<InstructionFingerprint>>
```
For each `act.define`:
1. Validate compute region (warn if boundary layout ops are found).
2. Compute fingerprint of core ops (yield, constants, tensor.empty filtered out).
3. Insert into catalog keyed by fingerprint hash.

### Match source ops
```
for each func.func in module:
  // Phase 1: try multi-op patterns first (largest match wins)
  for each multi-op instruction fingerprint (sorted by size, descending):
    scan source graph for matching subgraphs
    record match candidates for matched subgraphs

  // Phase 2: single-op matching for remaining unmatched ops
  for each unmatched linalg op in source function:
    key = computeFingerprint(op)
    candidates = catalog.lookup(key)
    for each candidate:
      if detailedMatch(op, candidate):
        record match candidate
```

### Match candidate structure
```
struct MatchCandidate {
  SmallVector<Operation*> sourceOps;    // one op for single-op, multiple for multi-op
  DefineOp instruction;                 // the matched act.define
  IRMapping binding;                    // source SSA values → instruction compute block args
  unsigned priority;                    // multi-op > single-op; more specific > less specific
};
```

## What gets matched, what doesn't

**Matched by Stage 1** (compute ops):
- `linalg.matmul`, `linalg.batch_matmul`, `linalg.contract`
- `linalg.conv_2d_nchw_fchw` and other conv variants
- `linalg.add`, `linalg.sub`, `linalg.mul`, `linalg.softmax`
- `linalg.generic` (elementwise, reductions, etc.)
- `linalg.map`, `linalg.reduce`, `linalg.transpose` (when used as compute)
- `linalg.broadcast` (when part of a fused pattern)

**Not matched by Stage 1** (infrastructure ops, handled by later stages):
- `tensor.empty` — allocation, no instruction needed
- `linalg.fill` — initialization, handled by data movement / accumulation patterns in Stage 2
- `tensor.pad` — padding, handled during tiling in Stage 2
- `arith.constant` — constants, folded during lowering

**Reported as unmatched**:
- Any compute op with no matching instruction → diagnostic: "no instruction found for `linalg.generic` at line X"
- This is informational, not a hard error (later stages or the user can decide what to do)

## Element type handling
For the MVP, require exact element type match between source ops and instruction compute regions. If the source has `f32` and the instruction has `bf16`, they don't match. A type conversion pre-pass is future work.

## Batch/rank handling
Stage 1 only matches when the iteration space structure (number of parallel/reduction dimensions) is the same. If the source has `linalg.batch_matmul` (3D) but the ISA only has `@gemm` (2D matmul), Stage 1 does NOT match them. The batch dimension loop is Stage 2's responsibility — it must peel off the batch dim and then Stage 1's match of the inner 2D matmul applies.

This means the pre-pass pipeline might also need a **batch dimension peeling** pass that converts `linalg.batch_matmul` into `scf.for` + `linalg.matmul` before Stage 1 runs. Alternatively, Stage 2 can handle this during tiling. TBD.

## Design note: boundary stripping as validator
The original Version 1 design used boundary stripping as a heuristic to extract fingerprints from mixed compute regions. After the compute/access separation refactor (see "Design Decision" in thoughts.md), compute regions are enforced to contain only computation. The boundary stripping logic is now repurposed as a **validator**: during catalog building, it scans each instruction's compute region and warns about any layout ops that should have been moved to the addr region. This catches ISA authoring errors early, while the fingerprint extraction itself is straightforward — just collect all non-infrastructure ops.
