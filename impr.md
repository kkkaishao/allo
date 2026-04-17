# Post-MVP Roadmap

After wiring the graph-based Stage 1 into the downstream pipeline, the next goal is not to add more ad hoc cases. The immediate goal is to make the current implementation structurally robust and consistent with the design direction in `thoughts.md` and `stage1.md` / `stage2.md` / `stage3.md`.

This roadmap is intentionally direct: the current implementation works for simple cases, but its architecture needs restructuring around the decision to **remove tiling from the backend**. Tiling is now a midend concern — the backend assumes well-tiled input and focuses on instruction selection, parameter solving, and code emission.

The roadmap below focuses on stabilizing the pipeline around explicit planning data, structured layout reasoning, and the new tiling-free backend design. It does **not** aim to turn the compiler into a general graph optimizer.

> **Note on ISA construction**: the current ISAs remain too simple to validate the next phases thoroughly. New ISAs should be added in `drafts/isa/` together with matching models in `drafts/models/`, especially for:
> - multi-op compute instructions
> - non-trivial relayout chains
> - mixed memory topologies
> - data-movement alternatives
> - sub-element operand modes

---

## Guiding principles

- **Compute/access separation remains the core rule.** Compute regions define semantics; addr regions define access, relayout, and buffer-side shape adaptation.
- **Tiling is a midend concern.** The backend assumes input programs are already well-tiled. Each linalg op must have shapes fitting within some instruction's native capacity.
- **The selected plan must become the single source of truth.** Stages must not rediscover graph structure from source ops after Stage 1.
- **Layout and access reasoning should be structural, not encoded in scattered special cases.**
- **The internal solver should stay intentionally small.** Easy cases should be solved internally; hard coupled cases should be detected cleanly and exported later.
- **Instruction selection should remain compute-first.** Data movement and resource choices exist to support selected compute instructions, not to replace them.

---

## Current diagnosis

The implementation has a useful baseline, but it needs restructuring:

1. **Tiling analysis is being removed.**
   - The previous `TilingAnalysis.h/cpp` mixed tiling concerns into the backend.
   - This created the graph-vs-anchor mismatch: Stage 1 was graph-based, Stage 2/3 were anchor-based.
   - Removing tiling eliminates this structural tension and simplifies the backend.

2. **`SemanticsGraph` edges are not yet the authoritative representation of value flow.**
   - Stage 1 builds selected graph edges.
   - The planner currently reconstructs most of the logical dataflow from boundary bindings instead of consuming graph edges directly.
   - This duplicates logic and makes layout propagation fragile.

3. **The planner exists, but it is not yet the full decision carrier.**
   - Some decisions live in Stage 1 matching results.
   - Some live in the param solution state.
   - Some are recomputed in Stage 3 resource and emission helpers.
   - This split makes re-selection, forwarding, and layout-sensitive reasoning harder than it should be.

4. **Selection happens too early.**
   - Stage 1 greedily selects non-overlapping matches before parameter solving and resource feasibility are known.
   - A locally larger semantic match can therefore block a later-feasible smaller decomposition.

5. **Layout/access reasoning is still too special-cased.**
   - Some transforms are carried on graph edges.
   - Some are interpreted only in planner/resource logic.
   - Some are handled only in emission-time offset computation.
   - This prevents the access side from becoming a stable abstraction.

6. **Offset computation needs modernization.**
   - The previous loop-IV-based dynamic offset computation is being replaced by affine map tracing via `AffineValueMapBuilder`.
   - This is more general and more principled, but not yet implemented.

---

## Recommended phases

## Phase A (IMMEDIATE): Remove tiling, establish new baseline

**Main goal**: Remove all tiling-related code from the backend and replace with parameter-solving-only logic.

### Implementation steps

1. **Create `ParamSolving.h/cpp`.**
   - Extract shape-matching parts from `TilingAnalysis.cpp`: `extractSymbolicShapes`, `extractSymbolicIterationDomain`, `classifyAddrParams`.
   - New `solveParams` function: equate symbolic and source iteration domains, solve constraints, reject on mismatch (no tiling factors).
   - New `runParamSolving(SemanticsGraph&, ModuleOp)` entry point.

2. **Update `Planning.h/cpp`.**
   - Replace `const TilingAnalysis *tilingAnalysis` with `const ParamSolution *paramSolution` in `LogicalPlanNode`.
   - Update `buildLogicalPlan` to accept `GraphParamSolution`.
   - Update `evaluateOperandSlotCount` to use `ParamSolution::solvedParams`.
   - Remove `hasTiledReduction` and tiling-tied accumulator init logic.

3. **Simplify `CodeEmission.cpp`.**
   - Remove all `scf.for` loop generation.
   - Remove dynamic offset computation from loop IVs.
   - Keep flat `act.emit` emission with static offsets.
   - Accept `GraphParamSolution` instead of `GraphTilingAnalysis`.

4. **Update `ConvertCanonicalFormToAct.cpp`.**
   - Call `runParamSolving` instead of `runTilingAnalysis`.

5. **Delete `TilingAnalysis.h/cpp`.**

### Expected result

A working backend that matches and emits for programs where source shapes exactly fit instruction capacity. No loop generation. Clean rejection when shapes don't fit.

---

## Phase B (NEXT): Affine offset tracing

**Main goal**: Support programs with affine loop nests (from midend tiling) by tracing offsets from the source IR.

### Implementation steps

1. **Copy `AffineValueMapBuilder` into `act/Support/AffineMapComposer.{h,cpp}`.**
   - Remove allo-specific dependencies.
   - Keep core functionality: SSA tracing through arith ops, affine.apply composition.

2. **Per-region instruction selection.**
   - Walk function body; for ops inside affine loop bodies, run matching + param solving + emission within that region.

3. **Affine offset computation.**
   - For source operands from `tensor.extract_slice` (or similar), use `AffineValueMapBuilder` to compose offset expressions.
   - Emit `affine.apply` for dynamic offsets.
   - Static fallback when all operands are constant.

4. **Update emission to handle mixed static/dynamic addr params.**
   - Shape params: always static (from `ParamSolution`).
   - Offset params: static when no enclosing loops, dynamic via `affine.apply` otherwise.

### Expected result

The backend handles tiled programs (affine loops + small linalg ops) by tracing offsets from the source IR, without generating any tiling loops itself.

---

## Phase C1 (NEXT): Make the selected graph authoritative

**Main goal**: make `SemanticsGraph` and the planner carry the full selected compute/dataflow structure, instead of allowing later stages to reinterpret matches heuristically.

### Implementation steps

1. **Make `SemanticsGraph` edges first-class inputs to planning.**
   - Build `LogicalPlan` from `SemanticsGraph.nodes` **and** `SemanticsGraph.edges`.
   - Do not reconstruct node-to-node value flow only from boundary bindings.
   - Preserve producer output index, consumer input index, and transform chain as planner-level facts.

2. **Represent outputs explicitly as logical values with consumers.**
   - For each selected compute node output, create a logical value even if it is immediately consumed by another selected node.
   - Record the value produced and the transforms required by each consumer edge.

3. **Populate writeback information structurally.**
   - `LogicalPlanNodeOutput.writebackTargetValueId` and `writebackTransforms` should no longer remain mostly empty.
   - For purely internal values, keep the internal value flow explicit and avoid inventing HBM round-trips early.

4. **Define planner invariants clearly.**
   - Every planner input points to exactly one logical value.
   - Every planner output defines exactly one logical value.
   - Every logical value has one defining node or is a function input.
   - Every logical consumer use carries an explicit required transform chain.

5. **Add planner-level debug dumps that expose value flow directly.**

### Expected result

After this phase, Stage 3 should no longer need to rediscover value flow from source IR shape alone.

---

## Phase C2 (NEXT): Move from anchor-driven to compute-signature-driven interpretation

**Main goal**: make Stage 2 and Stage 3 consume an explicit per-node compute signature instead of re-deriving semantics from one chosen anchor op.

### Implementation steps

1. **Introduce a planner-visible compute signature for each selected node.**
   - Record which source op determines iteration structure.
   - Record operand-to-iteration mapping.
   - Record which outputs are externally visible vs internal subgraph values.

2. **Stop selecting the iteration-domain source op heuristically.**
   - Replace "pick the linalg op with the most loops" with an explicit compute anchor supplied by Stage 1.

3. **Make operand offset computation use planner-carried operand semantics.**
   - Offset computation driven by the selected node's operand-role mapping + logical value layout + required transform chain.

4. **Add validation for multi-op instruction nodes.**
   - Reject nodes whose compute signature cannot explain all matched source ops consistently.

### Expected result

Multi-op matches are interpreted consistently instead of being collapsed onto one anchor op.

---

## Phase C3 (LATER): Delay irreversible selection until feasibility is known

**Main goal**: avoid committing to a semantic match before parameter solving and resource feasibility have been checked.

### Implementation steps

1. **Change Stage 1 output from "final selection" to "candidate set per region/op group".**
2. **Introduce a lightweight selection pipeline**: candidate → param solving feasibility → resource feasibility → final plan.
3. **Keep search local and bounded**: best candidate, next-best non-overlapping, decomposition fallback.
4. **Deterministic tie-breaking**: prefer semantically larger matches, fewer movement steps, fewer transforms.
5. **Expose rejection reasons in debug output**: semantic mismatch, param solving failure, resource overflow, no movement path.

### Expected result

The compiler can recover from locally attractive but globally infeasible semantic matches.

---

## Phase D1 (LATER): Unify layout/access reasoning into explicit signatures

**Main goal**: replace scattered layout heuristics with a shared representation.

### Implementation steps

1. **Define a richer transform/signature representation** usable by matching, planning, resource allocation, and emission.
2. **Generalize beyond transpose-only movement signatures.**
3. **Represent access-side shape and offset semantics together** in a common "access signature".
4. **Handle `AddrParamKind::Mixed` explicitly.**
5. **Add `act.tiled` end-to-end support.**

### Expected result

Layout and access decisions become shared planner facts instead of emission-time guesses.

---

## Phase D2 (LATER): Make resource planning graph-aware

**Main goal**: make residence, forwarding, and movement planning operate on logical values and graph edges.

### Implementation steps

1. **Plan residences per logical value**, not only per instruction operand.
2. **Generalize forwarding beyond adjacent nodes** using value-based producer-consumer edges.
3. **Remove global mode decisions based on the first instruction.**
4. **Make movement planning path-based.**
5. **Preserve external-vs-internal value distinctions.**

### Expected result

Resource planning matches the value-centric structure, and Stage 3 emission becomes simpler.

---

## Phase E (LATER): Keep the internal solver small, but make its boundary explicit

**Main goal**: improve Stage 2 coverage without turning the compiler into a general solver.

### Implementation steps

1. **Validate symbolic consistency explicitly.**
2. **Support a few more common internal cases**: repeated constraints, GCD reduction, bounded multiplicative forms.
3. **Differentiate "unsupported by solver" from "truly infeasible".**
4. **Define an exportable constraint model.**
5. **Only add external solver integration after the constraint model stabilizes.**

### Expected result

The solver remains limited, but its failure modes become principled and extensible.

---

## Validation strategy

### Required test categories

1. **Parameter solving tests**
   - exact shape match (no params)
   - parametric shape match (single param, consistent constraints)
   - shape mismatch rejection (source exceeds instruction capacity)
   - inconsistent param constraints rejection

2. **Affine offset tracing tests**
   - static offsets (no enclosing loops)
   - single affine loop with constant stride
   - nested affine loops with composed offsets
   - offsets through `tensor.extract_slice` chains

3. **Multi-op semantic match tests**
   - fused instruction with non-anchor internal op
   - match success and failure under induced-exact constraints

4. **Planner graph tests**
   - internal selected-node edge propagation
   - external boundary input/output handling
   - writeback target and transform propagation

5. **Layout/access tests**
   - transpose + slice chains
   - mixed transform chains across selected-node boundaries
   - `act.tiled` symbolic handling

6. **Resource planning tests**
   - mixed single-buffer and multi-buffer instructions
   - forwarding across boundaries
   - explicit writeback versus internal-only value flow

---

## Summary

The main architectural change is removing tiling from the backend. This eliminates the graph-vs-anchor mismatch that was the primary source of brittleness, and simplifies the pipeline:

- Stage 2 shrinks from tiling analysis to parameter solving
- Stage 3 drops loop generation and uses affine offset tracing instead
- The backend becomes a clean instruction selector + emitter

The remaining structural work (making the planner authoritative, compute signatures, deferred selection, unified layout reasoning) proceeds on a cleaner foundation.
