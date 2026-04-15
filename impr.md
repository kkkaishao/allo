# Post-MVP Roadmap

After wiring the graph-based Stage 1 into the downstream pipeline, the next goal is not to add more ad hoc cases. The immediate goal is to make the current implementation structurally robust and consistent with the design direction in `thoughts.md` and `stage1.md` / `stage2.md` / `stage3.md`.

This roadmap is intentionally direct: the current implementation works for simple cases, but its architecture is still fragile in several important ways. In particular, Stage 1 now produces graph-shaped matches, while Stage 2 and Stage 3 still interpret those matches mostly through a single-op, anchor-driven model. That mismatch is now the main source of brittleness.

The roadmap below focuses on stabilizing the pipeline around explicit planning data, structured layout reasoning, and limited-but-clean solver behavior. It does **not** aim to turn the compiler into a general graph optimizer.

> **Note on ISA construction**: the current ISAs remain too simple to validate the next phases thoroughly. New ISAs should be added in `drafts/isa/` together with matching models in `drafts/models/`, especially for:
> - multi-op compute instructions
> - non-trivial relayout chains
> - mixed memory topologies
> - data-movement alternatives
> - sub-element operand modes

---

## Guiding principles

- **Compute/access separation remains the core rule.** Compute regions define semantics; addr regions define access, relayout, and buffer-side shape adaptation.
- **The selected plan must become the single source of truth.** Stages must not rediscover graph structure from source ops after Stage 1.
- **Layout and access reasoning should be structural, not encoded in scattered special cases.**
- **The internal solver should stay intentionally small.** Easy cases should be solved internally; hard coupled cases should be detected cleanly and exported later.
- **Instruction selection should remain compute-first.** Data movement and resource choices exist to support selected compute instructions, not to replace them.

---

## Current diagnosis

The current implementation has a useful baseline, but it is still fragile in the following ways:

1. **Stage 1 is graph-based, but Stage 2 and Stage 3 are still anchor-based.**
   - Stage 1 can now select a multi-op instruction match.
   - Stage 2 still derives iteration structure from a single chosen linalg op.
   - Stage 3 still computes operand offsets and loop structure from a single anchor-style interpretation.
   - This is acceptable for anchor-dominant cases, but not robust for general multi-op instructions.

2. **`SemanticsGraph` edges are not yet the authoritative representation of value flow.**
   - Stage 1 builds selected graph edges.
   - The planner currently reconstructs most of the logical dataflow from boundary bindings instead of consuming graph edges directly.
   - This duplicates logic and makes layout propagation fragile.

3. **The planner exists, but it is not yet the full decision carrier.**
   - Some decisions live in Stage 1 matching results.
   - Some live in Stage 2 tiling state.
   - Some are recomputed in Stage 3 resource and emission helpers.
   - This split makes re-selection, forwarding, and layout-sensitive reasoning harder than it should be.

4. **Selection happens too early.**
   - Stage 1 greedily selects non-overlapping matches before tiling and resource feasibility are known.
   - A locally larger semantic match can therefore block a later-feasible smaller decomposition.

5. **Layout/access reasoning is still too special-cased.**
   - Some transforms are carried on graph edges.
   - Some are interpreted only in planner/resource logic.
   - Some are handled only in emission-time offset computation.
   - This prevents the access side from becoming a stable abstraction.

6. **The Stage 2 solver is weaker than the design intent.**
   - Constraint consistency is not always validated explicitly.
   - Multi-constraint and coupled-parameter cases are still mostly rejected instead of degraded gracefully.
   - Unsupported cases are not yet surfaced through a clean “internal solver limit reached” boundary.

---

## Recommended phases

## Phase C1 (NEXT): Make the selected graph authoritative

**Main goal**: make `SemanticsGraph` and the planner carry the full selected compute/dataflow structure, instead of allowing later stages to reinterpret matches heuristically.

### Implementation steps

1. **Make `SemanticsGraph` edges first-class inputs to planning.**
   - Build `LogicalPlan` from `SemanticsGraph.nodes` **and** `SemanticsGraph.edges`.
   - Do not reconstruct node-to-node value flow only from boundary bindings.
   - Preserve producer output index, consumer input index, and transform chain as planner-level facts.

2. **Represent outputs explicitly as logical values with consumers.**
   - For each selected compute node output, create a logical value even if it is immediately consumed by another selected node.
   - Record both:
     - the value produced at the selected node boundary
     - the transforms required by each consumer edge

3. **Populate writeback information structurally.**
   - `LogicalPlanNodeOutput.writebackTargetValueId` and `writebackTransforms` should no longer remain mostly empty.
   - For outputs that eventually materialize back to an external logical value, record the writeback target explicitly in the planner.
   - For purely internal values, keep the internal value flow explicit and avoid inventing HBM round-trips early.

4. **Define planner invariants clearly.**
   - Every planner input points to exactly one logical value.
   - Every planner output defines exactly one logical value.
   - Every logical value has one defining node or is a function input.
   - Every logical consumer use carries an explicit required transform chain.

5. **Add planner-level debug dumps that expose value flow directly.**
   - Dump logical values, defining node, consumer uses, and transform chains.
   - Dump selected graph edges and their corresponding logical-plan edges side by side.

### Expected result

After this phase, Stage 3 should no longer need to rediscover value flow from source IR shape alone. The selected graph becomes the canonical source of compute/dataflow truth.

---

## Phase C2 (NEXT): Move from anchor-driven interpretation to compute-signature-driven interpretation

**Main goal**: make Stage 2 and Stage 3 consume an explicit per-node compute signature instead of re-deriving semantics from one chosen anchor op.

### Implementation steps

1. **Introduce a planner-visible compute signature for each selected node.**
   - Record which source op determines iteration structure.
   - Record operand-to-iteration mapping for the selected instruction instance.
   - Record which outputs correspond to externally visible results versus internal subgraph values.

2. **Stop selecting the iteration-domain source op heuristically in Stage 2.**
   - Remove the “pick the linalg op with the most loops” rule from the tiling path.
   - Replace it with an explicit compute anchor / compute signature supplied by Stage 1 or planner construction.
   - Validate that the selected compute signature is consistent with the instruction node and the matched source subgraph.

3. **Make operand offset computation use planner-carried operand semantics.**
   - Stage 3 should not assume all operand offsets can be derived from the anchor op’s indexing maps.
   - Offset computation should be driven by:
     - the selected node’s operand-role mapping
     - the logical value layout at the current residence
     - the required transform chain on the incoming edge

4. **Separate “semantic anchor” from “tiling anchor” if necessary.**
   - In some fused instructions, the op that best identifies the pattern is not the same op that should determine loop bounds.
   - Allow the planner to carry both roles explicitly if they diverge.

5. **Add validation for multi-op instruction nodes.**
   - Reject selected nodes whose planner-visible compute signature cannot explain all matched source ops consistently.
   - Fail early and clearly rather than letting later emission silently collapse back to single-op behavior.

### Expected result

After this phase, multi-op matches are interpreted consistently by Stage 2 and Stage 3 instead of being implicitly collapsed onto one anchor op.

---

## Phase C3 (NEXT): Delay irreversible selection until feasibility is known

**Main goal**: avoid committing to a semantic match before tiling, layout, and resource feasibility have been checked.

### Implementation steps

1. **Change Stage 1 output semantics from “final selection” to “candidate set per region/op group”.**
   - Stage 1 should still rank candidates, but not permanently discard alternatives too early.
   - Preserve enough information to revisit a smaller or simpler match when a larger one becomes infeasible.

2. **Introduce a lightweight selection pipeline in the planner.**
   - Candidate semantic match
   - Tiling feasibility
   - Resource/layout feasibility
   - Final chosen plan

3. **Keep the search local and bounded.**
   - Do not introduce global combinatorial search yet.
   - Restrict fallback to:
     - best candidate
     - next-best non-overlapping candidate(s)
     - decomposition into smaller selected nodes where already available

4. **Define deterministic tie-breaking.**
   - Prefer:
     - semantically larger matches if feasible
     - fewer total tiling iterations
     - fewer movement steps
     - fewer extra layout transforms
   - Use stable source order and instruction name as final tie-breakers.

5. **Expose rejection reasons in debug output.**
   - Distinguish:
     - semantic mismatch
     - tiling infeasible
     - layout unsupported
     - resource overflow
     - no valid movement path

### Expected result

After this phase, the compiler can recover from locally attractive but globally infeasible semantic matches.

---

## Phase D1 (NEXT): Unify layout/access reasoning into explicit signatures

**Main goal**: replace scattered layout heuristics with a shared representation that can be consumed by matching, planning, resource allocation, and emission.

### Implementation steps

1. **Define a richer transform/signature representation.**
   - Extend the current transform chain so it can serve as the common representation for:
     - Stage 1 edge semantics
     - planner value-flow requirements
     - layout-aware data movement selection
     - HBM offset adaptation
   - Keep the representation structural and finite.

2. **Generalize beyond transpose-only movement signatures.**
   - Current data movement selection should not only match transpose.
   - Extract and compare richer access/layout signatures from identity instructions and planner edges.

3. **Represent access-side shape and offset semantics together.**
   - Today shape extraction and offset computation are split across different helpers.
   - Introduce a common “access signature” concept that describes:
     - visible tensor shape
     - logical layout transform
     - base-offset semantics
     - slot-count semantics

4. **Handle `AddrParamKind::Mixed` explicitly.**
   - Mixed parameters should no longer fall back to mostly offset-like treatment.
   - Their role in both shape solving and address generation should be modeled directly.

5. **Add `act.tiled` end-to-end support.**
   - Support it in symbolic shape extraction.
   - Support it in slot-count planning.
   - Support it in emission-time offset generation.

### Expected result

After this phase, layout and access decisions become shared planner facts instead of emission-time guesses.

---

## Phase D2 (NEXT): Make resource planning graph-aware instead of instruction-local

**Main goal**: make residence, forwarding, and movement planning operate on logical values and graph edges rather than only on consecutive instruction entries.

### Implementation steps

1. **Plan residences per logical value, not only per instruction operand.**
   - A value may live in HBM, scratchpad, or both temporarily.
   - The planner should track current and desired residences explicitly.

2. **Generalize forwarding beyond adjacent nodes.**
   - Current forwarding is mostly adjacency-based.
   - Replace it with value-based forwarding opportunities derived from logical producer-consumer edges and compatible buffer/layout requirements.

3. **Remove global mode decisions based on the first instruction.**
   - Do not decide “single-buffer vs multi-buffer” from the first planned node.
   - Infer per-node and per-value movement needs from actual buffer topology and value-flow requirements.

4. **Make movement planning path-based.**
   - For a source value and required destination residence/layout, plan a movement path explicitly.
   - Keep the first version bounded to short paths, but make the representation path-shaped from the start.

5. **Preserve external-vs-internal value distinctions.**
   - Internal subgraph values should not be forced into HBM layouts unless required by later consumers or buffer constraints.
   - Writeback should be an explicit planner decision, not an implicit default.

### Expected result

After this phase, resource planning will match the value-centric structure already described in `thoughts.md`, and Stage 3 emission will become substantially simpler.

---

## Phase E (LATER): Keep the internal solver small, but make its boundary explicit

**Main goal**: improve Stage 2 coverage without turning the compiler into a general symbolic or integer solver.

### Implementation steps

1. **Validate symbolic consistency explicitly.**
   - If multiple operands constrain the same iteration dimension, check equivalence directly.
   - Treat ISA-side inconsistency as an early hard error.

2. **Support a few more common internal cases.**
   - repeated single-parameter constraints
   - simple min/GCD-style reductions for repeated constraints
   - bounded handling of common multiplicative forms where one variable is already fixed

3. **Differentiate “unsupported by internal solver” from “truly infeasible”.**
   - Unsupported should be surfaced clearly so it can later become an external-solver export case.
   - Infeasible should remain a semantic/planning rejection.

4. **Define an exportable constraint model.**
   - Keep the internal representation explicit enough to serialize:
     - symbolic shape equalities
     - divisibility constraints
     - padding requirements
     - access/layout compatibility constraints

5. **Only add external solver integration after the constraint model stabilizes.**
   - Export first.
   - Integrate later.

### Expected result

The solver remains intentionally limited, but its failure modes become principled and extensible.

---

## Validation strategy

The next iterations should be validated with focused tests that target structural fragility, not only end-to-end happy paths.

### Required test categories

1. **Multi-op semantic match tests**
   - fused instruction with non-anchor internal op
   - fused instruction whose output role is not trivially the anchor output
   - match success and failure under induced-exact constraints

2. **Planner graph tests**
   - internal selected-node edge propagation
   - external boundary input/output handling
   - writeback target and transform propagation

3. **Selection fallback tests**
   - larger semantic match that is tiling-infeasible
   - smaller fallback decomposition that is feasible
   - deterministic re-selection

4. **Layout/access tests**
   - transpose + slice chains
   - static slice propagation through planning and movement selection
   - mixed transform chains across selected-node boundaries
   - `act.tiled` symbolic handling

5. **Resource planning tests**
   - mixed single-buffer and multi-buffer instructions in one function
   - forwarding across more than one local boundary
   - explicit writeback versus internal-only value flow

6. **Solver boundary tests**
   - consistent repeated constraints
   - inconsistent ISA-side constraints
   - internally unsupported but conceptually valid coupled constraints

---

## Summary

The main issue is no longer “missing features” in isolation. The main issue is that the pipeline now has two competing internal models:

- a graph-shaped semantic-selection model
- a mostly single-op downstream interpretation model

The next work should therefore focus on making the planner the single structural bridge between those stages, turning graph edges and layout requirements into first-class planning data, and delaying irreversible selection until feasibility is known.

That is the shortest path from “working on simple cases” to “robust enough to extend without accumulating brittle heuristics”.
