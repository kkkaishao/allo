# Stage 2: Parameter Solving

## Scope

Stage 2 takes the `SemanticGraph` selected by Stage 1 and solves concrete
instruction addr parameters that are determined by source tensor shapes. It does
not tile, pad, generate loops, or choose alternate instructions. A mismatch is a
hard failure for the current selected graph.

Implementation files:

- `mlir/include/act/Support/SymbolicExpr.h`
- `mlir/lib/act/Support/SymbolicExpr.cpp`
- `mlir/include/act/Support/ParamSolving.h`
- `mlir/lib/act/Support/ParamSolving.cpp`

## Data Model

Stage 2 produces one `ParamSolution` per selected semantic node:

```cpp
struct ParamSolution {
  SemanticGraphNode &node;
  InstructionParamModel model;
  DenseMap<unsigned, int64_t> solvedParams;
  bool isValid = false;
};
```

The `InstructionParamModel` describes every access operand of the selected
instruction:

```cpp
struct SymbolicRegion {
  SymShape basis;
  SymShape counts;
  SymShape strides;
};

struct SymbolicAccess {
  unsigned operandIdx;
  StringAttr bufferName;
  BufferTypeInterface bufferType;
  AccessRole role;
  SymbolicRegion storage;
  SymShape visibleShape;
};
```

`visibleShape` is the tensor shape seen by the compute region. `storage` is the
underlying strided region used later for scratch allocation and address binding.

Addr params are classified as:

- `Shape`: appears in counts or relayout output shape.
- `Offset`: appears in strided basis or stride.
- `Mixed`: appears in both categories.

Only `Shape` and `Mixed` params may be solved from shape constraints. Pure
`Offset` params are left for Stage 3 placement or movement planning.

## Symbolic Expressions

`SymExpr` is a small integer expression tree over addr block arguments:

- `Constant`
- `Param`
- `Add`
- `Mul`

`buildSymExpr` supports:

- Integer attributes in `OpFoldResult`.
- Addr block arguments, mapped by argument number.
- Integer `arith.constant`.
- `arith.addi`.
- `arith.muli`.

Other producers are unsupported.

## Building an InstructionParamModel

`buildInstructionParamModel(defineOp, module)` performs these steps:

1. Resolve every source and destination buffer symbol to a `DeclareBufferOp`.
2. Assign access roles: sources are `Read`, destinations are `Write`.
3. Walk every access operand yielded by the instruction addr region.
4. Build the operand `visibleShape` with `generateShapeExpr`.
5. Build the operand `storage` with `generateStorageRegion`.
6. Classify addr params by scanning strided and expand-shape operands.

The number of addr yield operands is assumed to match the number of
source-plus-destination buffers; the verifier should have checked this before
Stage 2.

## Shape Semantics

`generateShapeExpr(accessOp, bufferType)` is recursive.

`act.strided`:

- Shape starts with the symbolic `counts`.
- For non-HBM buffers, the buffer element shape is appended.
- If the resulting rank is greater than one and the leading dim is constant `1`,
  that leading dim is dropped.

`act.expand_shape`:

- The visible shape is the op's symbolic `output_shape`.
- The source is still recursively checked enough to build the chain, but the
  output shape controls the visible rank.

`act.collapse_shape`:

- Recursively gets the source shape.
- Multiplies dimensions according to reassociation groups.

`act.transpose`:

- Recursively gets the source shape.
- Reorders dimensions by the permutation.

`act.tiled`:

- Unsupported by the current symbolic extractor.

`generateStorageRegion(accessOp)` only materializes the underlying strided
storage:

- `act.strided` contributes symbolic basis, counts, and strides.
- `act.expand_shape`, `act.collapse_shape`, and `act.transpose` recurse to their
  source and preserve the same storage region.
- `act.tiled` is unsupported.

## Collecting Source Constraints

Stage 2 collects shape constraints from `SemanticInputBinding`. This includes
DPS output operands, because linalg outs are operands of the source semantic op.

For every binding:

1. Read the bound source SSA value type.
2. Require a ranked tensor with static shape.
3. Associate that concrete shape with the instruction access operand index.

After solving the listed constraints, every instruction access operand must have
been constrained. If any source or destination access operand lacks a source
shape constraint, solving fails.

## Constraint Solving

For each constrained operand, Stage 2 equates:

```text
symbolic visibleShape dim == concrete source tensor dim
```

The solver is intentionally direct:

- `Constant == N`: succeeds only when equal.
- `Param == N`: binds the param to `N`, unless it is a pure offset param.
- `Add == N`: if one side is already known, solve the other side.
- `Mul == N`: if one side is already known, require exact divisibility and solve
  the other side.
- Expressions with two unknown sides, such as `p0 + p1 == N` or
  `p0 * p1 == N`, are unsupported and fail.

Repeated bindings for the same param must agree exactly. Inconsistent constraints
fail loudly.

## Output Contract

On success, `runParamSolving` returns:

- The instruction's full symbolic access model.
- Concrete bindings for shape-controlled addr params.
- `isValid = true`.

Offset params are expected to be bound later by Stage 3 if they correspond to
scratch placement or movement endpoints. Code emission ultimately requires every
addr param of every emitted instruction to be bound to a static integer.

## Examples

Fixed shape:

```text
visibleShape = [64, 64]
sourceShape  = [64, 64]
solvedParams = {}
```

Parametric square matmul:

```text
visibleShape = [Param<3>, Param<3>]
sourceShape  = [8, 8]
solvedParams = {3 -> 8}
```

Rejected mismatched square matmul:

```text
visibleShape constraints:
  Param<3> == 16
  Param<3> == 8
```

The same param is constrained to two different values, so solving fails.

## Limitations

- No tiling or padding. Shape mismatch means rejection.
- No fallback to another Stage 1 candidate after solving fails.
- Static tensor shapes are required.
- Only simple add/mul expressions with one known side are solved.
- `act.tiled` is unsupported in symbolic shape/storage extraction.
