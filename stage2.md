# Stage 2: Parameter Solving — Detailed Design

## Status

**Iteration 1 (same-rank param solving): DONE**
**Iteration 2 (input validation): PLANNED**

Tiling analysis has been fully removed from the backend. Stage 2 is now parameter solving only: given a semantic match from Stage 1, determine the concrete addr parameter values that make the source shapes fit the instruction. Matches are rejected when shapes don't fit — the midend/user is responsible for tiling.

Implementation:
- `mlir/include/act/Support/ParamSolving.h` — `AddrParamKind`, `ParamSolution`, `GraphParamSolution`, `runParamSolving`
- `mlir/lib/act/Support/ParamSolving.cpp` — symbolic shape extraction, constraint solving, param classification

Preserved infrastructure:
- `mlir/include/act/Support/SymbolicExpr.h` — `SymExpr` struct, `buildSymExpr`, `generateShapeExpr`, `SymShape`
- `mlir/lib/act/Support/SymbolicExpr.cpp` — SSA tracing, per-op shape generation (StridedOp, ExpandShapeOp, CollapseShapeOp, TransposeOp)

Removed:
- `mlir/include/act/Support/TilingAnalysis.h`
- `mlir/lib/act/Support/TilingAnalysis.cpp`
- `TilingScheme`, `DimTiling`, `GraphTilingAnalysis` structs
- Tiling factor computation, rank-mismatch suffix matching, contiguity validation
- `scf.for` loop generation in CodeEmission
- Dynamic offset computation from loop IVs
- Accumulator init logic for tiled reductions
- `MLIRSCFDialect` dependency

## Goal

For each match candidate from Stage 1, solve for the instruction's addr parameters that make the source operand shapes fit. Reject matches where shapes are incompatible (the midend should have tiled them). Output solved parameter values and parameter classifications per matched node.

## Architectural decision: No tiling in the backend

The previous design mixed tiling analysis into the backend (Stage 2 computed tile factors, Stage 3 generated `scf.for` loops). This is removed for the following reasons:

1. **Tiling is a midend concern.** The backend's job is instruction selection and code emission for an already-tiled computation. By the time the computation reaches the backend, each linalg op should already have shapes that fit within some instruction's native capacity.

2. **The tiling analysis was the weakest link.** It was the source of the graph-vs-anchor mismatch diagnosed in `impr.md`: Stage 1 produces graph-shaped matches while Stage 2 and Stage 3 interpreted them through a single-op anchor model. Removing tiling eliminates this structural tension.

3. **It simplifies the backend dramatically.** Without tiling, the backend doesn't need to generate `scf.for` loops, classify dims as parallel/reduction, compute dynamic offsets from loop IVs, handle accumulator initialization for tiled reductions, or validate tile contiguity.

**What users must provide:** Input programs must be well-tiled such that each linalg op can be covered by exactly one instruction invocation. This is a contract with the midend/user.

**Potential improvement (deferred):** A validation pass that checks whether input programs are instruction-selectable before running the full pipeline. This would report which ops cannot be covered and why (shape mismatch, no matching instruction, etc.).

## Input / Output

- **Input**: `SemanticsGraphs` from Stage 1 (matched source ops to instruction definitions), plus buffer/ISA declarations in the module.
- **Output**: `SmallVector<GraphParamSolution>` — per-graph, per-node solved parameter values and classifications.

---

## Data Structures

### ParamSolution (replaces TilingScheme + TilingAnalysis)

```cpp
/// Classification of addr parameters.
enum class AddrParamKind {
  Shape,  // appears in counts/output_shape — controls computation size
  Offset, // appears in basis — controls data position
  Mixed,  // appears in both
};

/// Solved parameters for one matched node.
struct ParamSolution {
  SemanticsGraphNode *node = nullptr;

  /// Solved shape params: addr block arg index -> concrete value.
  DenseMap<unsigned, int64_t> solvedParams;

  /// Param classifications.
  DenseMap<unsigned, AddrParamKind> paramKinds;

  /// Whether the solution is valid (all shapes match).
  bool isValid = false;

  DefineOp getInstruction() const;
};

using GraphParamSolution = SmallVector<ParamSolution, 4>;
```

### Symbolic Expression Framework (unchanged)

The `SymExpr` infrastructure is preserved from the previous design. See the Symbolic Expression Framework section below for reference.

---

## Algorithm

### Step 1: Extract instruction's symbolic shapes

For each matched instruction, walk all yield operands in the addr region:

```
for i in 0..numOperands:
  yieldOperand = addrBlock.getTerminator()->getOperand(i)
  bufferType = getBufferTypeForOperand(defineOp, i)
  symShapes[i] = generateShapeExpr(yieldOperand, bufferType)
```

This is identical to the previous design — reuses `extractSymbolicShapes`.

### Step 2: Map symbolic shapes to iteration domain

Use the compute region's linalg op indexing maps to relate operand dimensions to iteration dimensions:

```
For @gemm:
  compute has linalg.matmul with maps: A->(M,K), B->(K,N), C->(M,N)
  symShapes: A=[64,64], B=[64,64], C=[64,64]
  -> iteration bounds: M=64, N=64, K=64
```

For each iteration dimension `d`, collect the symbolic expression from the first operand that constrains it. Multiple constraints on the same dimension should be consistent (error if not — indicates ISA authoring bug).

This is identical to the previous design — reuses `extractSymbolicIterationDomain`.

### Step 3: Compare against source iteration domain

Get concrete iteration domain bounds from the matched source linalg op via `getStaticLoopRanges()`.

**Key difference from previous design:** Instead of computing tiling factors when source exceeds instruction capacity, we **reject** the match:

```
for each iteration dimension d:
  native_bound = evaluate(symbolicIterDomain[d], solvedParams)
  source_bound = sourceIterDomain[d]
  if source_bound != native_bound:
    return reject("shape mismatch: source dim " + d + " = " + source_bound +
                  " but instruction native bound = " + native_bound)
```

For parametric bounds (SymExpr is Param), we solve by direct assignment: `params[paramIdx] = source_bound`. For compound expressions, evaluate after params are solved.

### Step 4: Classify addr params

Classify each addr parameter as Shape, Offset, or Mixed by examining where it appears in the access chains. This is identical to the previous `classifyAddrParams`:

```
for each access op in addr block:
  for act.strided: basis params -> Offset, counts params -> Shape
  for act.expand_shape: output_shape params -> Shape
  for act.collapse_shape: (no new params)
  for act.transpose: (no new params)
```

### Step 5: Output

Return `ParamSolution` with `solvedParams`, `paramKinds`, and `isValid`.

---

## Symbolic Expression Framework (reference)

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

  int64_t evaluate(ArrayRef<int64_t> paramValues) const;
  void collectParams(DenseSet<unsigned> &out) const;
  SymExpr simplify() const;
  std::string toString(ArrayRef<StringRef> paramNames = {}) const;
  bool isConstant() const;
  bool isParam() const;
};
```

### `generateShapeExpr` — free function

```cpp
FailureOr<SymShape> generateShapeExpr(Operation *accessOp, BufferTypeInterface bufferType);
```

Dispatches via `isa<>` checks. Per-op behavior:

- **`act.strided`**: `shape = [counts...] ++ bufferType.getShape()`
- **`act.expand_shape`**: recursive source shape, replace per reassociation with output_shape dims
- **`act.collapse_shape`**: recursive source shape, merge per reassociation with `SymExpr::mul`
- **`act.transpose`**: recursive source shape, permute via `result[i] = sourceShape[perm[i]]`

---

## Constraint solving

The solver handles:

1. **Constant bounds**: direct comparison. Source must equal native.
2. **Single Param bounds**: assign `param = source_bound`. Check consistency if same param appears in multiple dimensions.
3. **Compound expressions**: evaluate after single-param constraints are solved. If the evaluated native bound doesn't match source, reject.

The solver is intentionally simple. Complex coupled-parameter cases are rejected cleanly rather than solved approximately.

### Solver boundary

If the internal solver cannot handle a constraint system (e.g., `Param(0) * Param(1) = N`), it rejects with a clear "unsupported constraint form" diagnostic. This is distinct from "infeasible" (where constraints are provably unsatisfiable).

Future work: export unsupported constraint systems to an external solver (Z3, OR-Tools).

---

## Implementation Plan

### Iteration 1: Parameter solving (replaces tiling analysis) — DONE

**Goal:** Replace `TilingAnalysis` with `ParamSolving` — same symbolic shape extraction, but reject instead of tile when shapes don't fit.

**What was done:**

1. Created `ParamSolving.h/cpp` with `ParamSolution`, `GraphParamSolution`, `runParamSolving`. Reused `extractSymbolicShapes`, `extractSymbolicIterationDomain`, `classifyAddrParams` verbatim. New `solveParams` rejects on shape mismatch (no tiling factors). New `solveNode` requires exact rank match (no suffix matching).

2. Updated `Planning.h/cpp`: `LogicalPlanNode.paramSolution` replaces `tilingAnalysis`. `evaluateOperandSlotCount` takes `DenseMap<unsigned, int64_t>` directly. Deleted `hasTiledReduction` and the accumulator init block in `buildResourcePlan`.

3. Updated `CodeEmission.h/cpp`: removed all `scf.for` loop generation, `LoopDim` struct, dynamic offset computation (`emitDataMovementDynamic`, `emitStoreDynamic`), accumulator init phase. `computeHBMOffset` simplified to always-static (returns `int64_t` instead of `HBMOffset` struct). Removed `MLIRSCFDialect` dependency.

4. Updated `ConvertCanonicalFormToAct.cpp`: calls `runParamSolving` instead of `runTilingAnalysis`.

5. Deleted `TilingAnalysis.h/cpp`, removed from `CMakeLists.txt`.

**Verified with:**
- `mm_bf16.mlir` + `qkv.mlir` — fixed 64x64, exact match, correct `act.emit` output
- `chain_mm_bf16.mlir` + `qkv.mlir` — two-node graph, correct chained emission
- `square_mm.mlir` + `tpu.mlir` — parametric `%size=8` solved correctly
- `batch_mm.mlir` + `tpu.mlir` — rank mismatch correctly rejected (Stage 1 filters it)

### Iteration 2: Input validation pass (potential improvement)

**Goal:** A diagnostic pass that checks whether each linalg op in the input program can be covered by some instruction in the ISA, without transforming the IR.

#### Step 1: Validation logic
- For each linalg op in the program, try all instructions: extract symbolic shapes, attempt constraint solving
- Report which ops have no valid instruction match and why (shape mismatch, no matching fingerprint, unsupported constraint)

#### Step 2: Integration
- Run as an optional early diagnostic before the full pipeline
- Does not block compilation — just emits warnings/errors

---

## Worked Examples

### Example 1: Fixed-shape instruction (no params to solve)

```
Source: linalg.matmul ins(tensor<64x64xbf16>, tensor<64x64xbf16>) outs(tensor<64x64xbf16>)
Instruction: @gemm from qkv.mlir (fixed 64x64)
```

Symbolic iteration domain: M=64, N=64, K=64 (all constant).
Source iteration domain: M=64, N=64, K=64.

All dimensions match exactly. solvedParams = {} (no shape params). isValid = true.

### Example 2: Parametric instruction (single param)

```
Source: linalg.matmul ins(tensor<8x8xf32>, tensor<8x8xf32>) outs(tensor<8x8xf32>)
Instruction: @matmul from tpu.mlir (%size param)
```

Symbolic iteration domain: M=Param(3), N=Param(3), K=Param(3).
Source iteration domain: M=8, N=8, K=8.

Constraints: Param(3)=8 (consistent across all dims). solvedParams = {3: 8}. isValid = true.

### Example 3: Shape mismatch (rejected)

```
Source: linalg.matmul ins(tensor<128x128xf32>, ...) outs(tensor<128x128xf32>)
Instruction: @gemm from qkv.mlir (fixed 64x64)
```

Symbolic iteration domain: M=64, N=64, K=64.
Source iteration domain: M=128, N=128, K=128.

Mismatch on every dimension. **Rejected** — the midend should have tiled the source to 64x64 before reaching the backend.

### Example 4: Inconsistent param constraints (rejected)

```
Source: linalg.matmul ins(tensor<16x8xf32>, tensor<8x12xf32>) outs(tensor<16x12xf32>)
Instruction: @matmul from tpu.mlir (%size param, requires M=N=K=%size)
```

Constraints: Param(3)=16 (from M), Param(3)=8 (from K), Param(3)=12 (from N). Inconsistent. **Rejected** — this instruction requires square matrices.
