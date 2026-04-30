# Stage 1: Semantic Matching

## Scope

Stage 1 builds a complete semantic covering of a flat source `func.func` using
the available `act.define` instructions. It does not solve addr params, allocate
buffers, insert data movement, or emit `act.emit`. Its output is a
`SemanticGraph` that says which source semantic ops are covered by which
instruction pattern, plus boundary bindings from source SSA values to
instruction access operands.

Implementation files:

- `mlir/include/act/Support/SemanticMatching.h`
- `mlir/lib/act/Support/SemanticMatching.cpp`

The pass driver is `ConvertCanonicalFormToAct.cpp`. Before Stage 1, it optionally
inlines an ISA file and then runs `linalg-morph-ops` with `genericToNamed`,
`canonicalize`, and `cse`.

## Semantic and Non-Semantic Ops

The current primitive semantic op set is intentionally small:

- `linalg.generic`
- `linalg.softmax`
- `linalg.map`
- `linalg.reduce`
- `linalg.contract`
- `linalg.add`
- `linalg.matmul`

These ops become graph nodes. Everything else must either be layout/provenance
that can be traced across graph edges, or placeholder infrastructure.

Layout/provenance ops:

- `tensor.extract_slice`
- `tensor.insert_slice`
- `tensor.expand_shape`
- `tensor.collapse_shape`
- `linalg.transpose`

Placeholder ops:

- `tensor.empty`
- `arith.constant`
- `linalg.fill` when all fill inputs are constants or block arguments

Current behavior: `linalg.transpose` is treated as a layout transform, not as a
selectable compute op.

## ProgramGraph

`ProgramGraph::build(func)` scans the function entry block and creates one node
per primitive semantic op. For every operand of every semantic op, it calls
`resolveLayoutChains` to trace back through layout/provenance ops until it
reaches:

- another semantic op result,
- a block argument or other external value,
- or a placeholder value.

The resulting edge records a `LayoutChain`:

```cpp
struct LayoutChain {
  Value source;
  std::optional<StaticSlice> slice;
  std::optional<StaticSlice> targetSlice;
  SmallVector<Operation *, 4> layoutOps;
};
```

Important implementation details:

- Slices must be fully static.
- Rank-reducing slice provenance is unsupported.
- Slice overlap/subtraction is handled for `tensor.insert_slice`, so one logical
  edge may split into multiple layout chains.
- `tensor.expand_shape` and `tensor.collapse_shape` are tracked as layout ops,
  but slice provenance is not propagated through the reshape.
- Unsupported provenance ops cause matching to fail.

Function return operands are traced in the same way and become external output
edges.

## InstructionGraph

`InstructionGraph::build(defineOp)` applies the same graph construction to an
instruction compute region. Instructions whose compute region has no primitive
semantic op are ignored by Stage 1. This is how identity/load/store style
instructions stay out of semantic matching; Stage 3 later discovers them as data
movement instructions.

Each non-empty instruction graph chooses an anchor node. The current heuristic is
the node with the largest total number of in/out edges, with earlier nodes
winning ties.

## SemanticIdentity

Each primitive semantic op has a `SemanticIdentity` used for fast pruning and
exact comparison.

Hash key:

- Op name.
- Number of DPS inputs and inits for `linalg::LinalgOp`.
- Indexing maps and iterator types for `linalg::LinalgOp`.
- Softmax loop iterator types for `linalg.softmax`.
- Operand/result counts for other primitive ops.

Detailed match:

- Op name, operand count, and result count must match.
- Operand/result types must either be exactly equal, or both shaped with the same
  rank and element type. Dimension sizes are ignored at this stage.
- `linalg.generic` additionally requires equal indexing maps, iterator types,
  and structurally equal scalar bodies. Constants in the body are compared by
  value.
- Non-generic primitive ops currently match by the structural checks above; for
  most named ops, the name is the semantics.

## Matching Algorithm

Stage 1 first builds an `InstructionCollection`:

```cpp
DenseMap<hash_code, SmallVector<unsigned, 2>> byAnchorHash;
std::vector<InstructionGraph> instructions;
```

For each source node in the `ProgramGraph`:

1. Look up instruction graphs whose anchor hash matches.
2. Confirm anchor semantic identity.
3. Run backtracking graph matching from the anchor.
4. Choose the next pattern node by the number of already-mapped neighbors.
5. Candidate source nodes are sorted by graph degree.
6. `hasConsistentEdges` enforces induced-exact internal edges between already
   mapped nodes.

Current edge comparison checks producer result id and consumer operand id. It
does not yet compare transform-chain equivalence.

## Building the SemanticGraph

After collecting candidates, Stage 1 greedily selects non-overlapping matches in
candidate order. It then verifies that every primitive semantic op in the
function is covered. Partial lowering is rejected.

The selected matches are sorted by source program order and converted into a
`SemanticGraph`:

- One `SemanticGraphNode` per selected instruction.
- Internal edges between selected instruction nodes.
- `SemanticInputBinding` entries for instruction compute block arguments that
  correspond to source boundary values. DPS output operands are also represented
  here because they are linalg operands.
- `SemanticOutputBinding` entries for instruction results that feed another
  selected node or a function result.

Edges internal to a multi-op selected instruction are not emitted as
inter-instruction dependencies; they are already represented inside the selected
instruction pattern.

## Limitations

- The pass driver currently rejects non-flat functions before Stage 1. Matching
  inside regions or through control flow is unsupported. It may be evaluated
  later if a workload needs it, but it is not an assumed roadmap item.
- Candidate selection is greedy and not cost-based.
- Transform-chain matching is incomplete.
- The primitive semantic op set is small.
- If Stage 2 later rejects a selected candidate, the current pipeline does not
  backtrack to another Stage 1 candidate.
