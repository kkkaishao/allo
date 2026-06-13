---
title: Allo Scheduling
createdAt: 2026-06-13
order: 3
summary: Reference for the Allo schedule API — selection, transform primitives, composition, streaming, and export.
keywords: ["Allo", "Scheduling", "Schedule", "Transform", "MLIR"]
---

%toc%
# Scheduling

A `Schedule` decouples the *algorithm* (what a kernel computes) from the
*schedule* (how it is mapped to hardware). It operates on the MLIR module
produced by a kernel and builds a transform-dialect script against stable
operation and value references. Calling `kernel.schedule()` compiles the kernel
and returns a `Schedule` bound to it.

```python
from allo.lang import i32, kernel, range

@kernel
def top(A: i32[16], B: i32[16]):
    for i in range(16, name="i"):
        B[i] = A[i] + 1

s = top.schedule()
i = s.loop("i")
outer, inner = s.split(i, factor=4)
s.pipeline(inner, ii=1).apply()

print(s.payload)
```

You select operations, loops, and buffers with typed refs, apply schedule
primitives, then either inspect the transformed `payload` module or hand it to a
backend with `s.export(...)`. The recommended selection entry points are the
short aliases `s.loop(...)`, `s.loops(...)`, `s.op(...)`, and `s.buffer(...)`.

Loop names come from the frontend iterator name. Write
`for i in range(16, name="i")` to make a loop selectable as `s.loop("i")`. A
`grid(..., name="ij")` names the whole loop-like operation, not the individual
axes.

## Constructing a Schedule

For a frontend kernel, use `Kernel.schedule()`. A templated kernel must be
specialized first (e.g. `gemm[i32, 32]`).

```python
s = top.schedule()
```

A `Schedule` can also be built from a standalone MLIR module, text, or file.
These forms have no source kernel, so they support every transform and the
inspection helpers but cannot be exported to a backend.

```python
from allo.schedule import Schedule

s = Schedule.from_module(module)
s = Schedule.from_string(mlir_text)
s = Schedule.from_file("kernel.mlir")
```

| Field      | Meaning                                                                           |
| ---------- | --------------------------------------------------------------------------------- |
| `payload`  | The MLIR module being scheduled. Reading it first applies any pending transforms. |
| `snapshot` | Immutable view of operations and buffer values at the current state.              |
| `query`    | Low-level query object used by the selection aliases.                             |
| `dirty`    | Whether pending transforms have not been applied yet.                             |
| `kernel`   | The source kernel, or `None` for module/string/file schedules.                    |

## The Schedule Model

The scheduler has three cooperating pieces:

1. **Payload** — the MLIR module being transformed.
2. **Snapshot** — an immutable index of payload operations and buffer values,
   keyed by a stable schedule ID, name, kind, and structural path. Selection
   reads only from the snapshot.
3. **Transform script** — a module of MLIR transform-dialect operations that is
   accumulated by primitives and run against the payload by `apply()`.

On construction the scheduler stamps internal schedule IDs onto the payload and
collects a snapshot. Refs carry those IDs, which is how a Python ref reconnects
to a payload operation after a transform runs.

Primitives fall into two kinds:

- **Tagging primitives** attach schedule attributes without changing IR
  topology. They append to the transform script and return the schedule for
  chaining, but defer execution until `apply()`. Existing refs stay valid.
- **Structural primitives** change the loop nest or function structure. They
  apply immediately, rebuild the snapshot, and return refs for the new topology.
  Refs captured before the change become stale.

`apply()` verifies the pending script, runs it on a clone of the payload,
verifies the result, refreshes the snapshot, and starts a fresh script for the
next batch. Reading `payload` or `snapshot` while `dirty` triggers `apply()`
automatically.

```python
s.pipeline(loop, ii=2)   # queued, dirty == True
s.unroll(loop2, factor=4)
s.apply()                # runs the batch, dirty == False
```

## Selecting Targets

### Aliases

The aliases on `Schedule` resolve to a single ref (or a tuple, for `loops()`)
and raise a source-aware diagnostic if the name is missing or ambiguous.

| Alias                                             | Result             | Equivalent query                          |
| ------------------------------------------------- | ------------------ | ----------------------------------------- |
| `s.op(name, *, under=None, kind=None, path=None)` | one `OpRef`        | `s.query.op(...).one()`                   |
| `s.loop(name, *, under=None, path=None)`          | one `LoopRef`      | `s.query.loop(...).one()`                 |
| `s.loops(*names, under=None, path=None)`          | tuple of `LoopRef` | `s.query.loop(...).names(...)` / `.all()` |
| `s.buffer(name, *, under=None, path=None)`        | one `BufferRef`    | `s.query.buffer(...).one()`               |

`s.loops()` with no names returns every loop in the primary function;
`s.loops("i", "j")` returns exactly those named loops in that order.

```python
i = s.loop("i")
i, j = s.loops("i", "j")
all_loops = s.loops()

func = s.op("top")
B = s.buffer("B")
```

`under` scopes a lookup to operations nested under another op (by ref or name).
`path` selects a specific snapshot path. `kind` matches the MLIR operation name,
for example `affine.for` or `scf.for`; it is kept on `op`/`query.op` for
advanced use.

### Low-level query

The query methods return a `RefSelection`. Use them when you need `.first()`,
`.all()`, `kind=`, or `path=`.

| Method                                                     | Result                                       |
| ---------------------------------------------------------- | -------------------------------------------- |
| `query.op(name=None, *, under=None, kind=None, path=None)` | Select operations.                           |
| `query.loop(name=None, *, under=None, path=None)`          | Select loop-like operations.                 |
| `query.buffer(name=None, *, under=None, path=None)`        | Select buffer values (arguments or results). |

| `RefSelection` method | Result                                                |
| --------------------- | ----------------------------------------------------- |
| `.one()`              | Exactly one match, else raise (missing or ambiguous). |
| `.first()`            | The first match, raise only if none.                  |
| `.all()`              | All matches as a list (possibly empty).               |
| `.names(*names)`      | One match per name, in order.                         |

### Refs

Refs are lightweight immutable values:

| Ref         | Meaning                                                           |
| ----------- | ----------------------------------------------------------------- |
| `OpRef`     | Any operation.                                                    |
| `LoopRef`   | A loop-like operation (`scf.for`, `affine.for`, `scf.parallel`).  |
| `BufferRef` | A buffer value (memref) owned by an operation argument or result. |

Primitives accept refs, names, or iterables of refs/names where a multi-target
operation is meaningful. A name must resolve unambiguously.

## Generic Passes

These primitives run an MLIR pass over op targets. With no target they default
to the primary function. They are tagging primitives — chain them and call
`.apply()`.

| Primitive                                  | Optimization                                                                                                           |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `s.cse(targets=None)`                      | Common subexpression elimination.                                                                                      |
| `s.dce(targets=None)`                      | Dead code elimination.                                                                                                 |
| `s.licm(targets=None)`                     | Loop-invariant code motion.                                                                                            |
| `s.canonicalize(targets=None)`             | Canonicalization patterns.                                                                                             |
| `s.apply_patterns(patterns, targets=None)` | Apply named rewrite patterns. `patterns` is a name or iterable of names; currently only `"canonicalize"` is supported. |

```python
s.cse().dce().canonicalize().apply()
```

## Loop and Memory Tags

Tagging primitives annotate the IR for the backend. They do not change topology,
so refs captured beforehand remain valid.

### `s.pipeline()`

`s.pipeline(targets=None, *, ii=1)`

Marks loop targets for pipelining with the given initiation interval — the
backend overlaps successive iterations so a new iteration starts every `ii`
cycles. `ii` must be a positive integer (default `1`). Returns the schedule.

```python
s.pipeline(s.loop("i"), ii=2)
```

### `s.dataflow()`

`s.dataflow(targets=None)`

Tags a function for task-level parallelism (`#pragma HLS dataflow`). The Vitis
emitter turns the attribute into the pragma, so the function's top-level
statements — for example the PE invokes of a systolic array, or composed stage
kernels connected by streams — run as a concurrent dataflow network instead of
sequentially. Defaults to the primary function. Returns the schedule.

### `s.unroll()`

`s.unroll(targets=None, *, factor=0, tag_only=False)`

Unrolls loop targets to expose parallelism and reduce loop overhead. `factor` is
a non-negative integer; `factor=0` means full unroll. By default
(`tag_only=False`) the loop is **physically unrolled immediately**, followed by
canonicalize/CSE cleanup. With `tag_only=True` it only attaches an unroll
attribute and defers like the other tags. Returns the schedule.

```python
s.unroll(s.loop("k"), factor=4)              # physical unroll now
s.unroll(s.loop("k"), factor=4, tag_only=True).apply()  # attribute only
```

### `s.partition()`

`s.partition(targets, *, dim=0, kind=Complete, factor=0)`

Partitions a buffer across memory banks so multiple elements can be accessed in
the same cycle. `targets` (required) are buffer refs. `dim` is the dimension to
partition (`0` partitions all dimensions). `kind` is one of:

| Kind         | Meaning                          | `factor`      |
| ------------ | -------------------------------- | ------------- |
| `s.Complete` | Split into individual registers. | must be `0`   |
| `s.Block`    | Contiguous blocks.               | must be `> 0` |
| `s.Cyclic`   | Round-robin across banks.        | must be `> 0` |

Returns the schedule.

```python
A = s.buffer("A")
s.partition(A, dim=1, kind=s.Cyclic, factor=4)
s.apply()
```

## Loop Restructuring

Structural primitives. Each applies immediately and returns refs for the new
topology; refs from before the call go stale.

### `s.affine()`

`s.affine(targets=None) -> list[LoopRef]`

Raises loop targets to affine form (`scf.for` → `affine.for`), preserving names.
Defaults to all loops in the primary function. Returns
the raised loop refs.

### `s.split()`

`s.split(target=None, *, factor=1) -> (outer, inner)`

Strip-mines one loop into an outer loop over tiles and an inner loop within a
tile. `factor` is the inner trip count (positive, default `1`). Returns
`(outer, inner)`.

```python
i = s.loop("i")
outer, inner = s.split(i, factor=4)
s.pipeline(inner, ii=1).apply()
```

### `s.reorder()`

`s.reorder(targets) -> tuple[LoopRef, ...]`

Permutes a perfectly nested band of affine loops into the requested order — used
to change locality or move a reduction inward. Requires at least two **affine**,
unique loop targets. Returns the loop refs in the requested order.

### `s.tile()`

`s.tile(targets=None, *, factors=1) -> (tiles, points)`

Tiles a loop band, producing an outer *tile* loop and an inner *point* loop per
axis — the combination of `split` across several loops with a `reorder`.
`factors` is a single int (broadcast to all loops) or one factor per loop.
Returns `(tile_loops, point_loops)` as two lists.

```python
tiles, points = s.tile(s.loops("i", "j"), factors=[2, 4])
s.pipeline(points[-1], ii=1).apply()
```

### `s.flatten()`

`s.flatten(targets) -> LoopRef`

Collapses two or more perfectly nested loops into a single loop, removing
nested-loop boundaries (often to enable a longer pipeline). Requires at least two
loop targets. Returns the flattened loop ref.

## Data Movement and Localization

Structural primitives that move computation or introduce on-chip buffers. They
apply immediately and return live refs.

### `s.compute_at()`

`s.compute_at(target, axis) -> LoopRef`

Fuses a producer operation into a consumer loop at the given affine `axis`,
interleaving their computation to shorten the producer's live range. The
producer loop nest is erased and its body moves under the axis loop. `axis` must
be an affine loop. Returns the live axis ref.

### `s.buffer_at()`

`s.buffer_at(target, axis) -> BufferRef`

Creates a localized buffer for `target` scoped to the affine loop `axis`,
staging data on-chip at that level of the nest. Returns the new buffer ref (a
`{base}.local` allocation).

### `s.reuse_at()`

`s.reuse_at(target, axis, *, ring=False) -> BufferRef`

Creates a reuse buffer at the affine loop `axis` that captures data reused across
iterations — the classic line/window buffer for stencils and convolutions. It is
smaller than `buffer_at` in steady state because it only holds the live reuse
window. Set `ring=True` to use a ring (circular) buffer. Returns the new buffer
ref (a `{base}.reuse` allocation).

## Outlining

### `s.outline()`

`s.outline(target, *, func_name, mapping=None) -> (kernel, call)`

Extracts an operation into its own function so it can be reused or scheduled
independently. With `mapping=None` it emits a `func.func` / `func.call` pair.
When `mapping` is an integer or a sequence of positive integers, it emits an
`allo.kernel` / `allo.invoke` pair with the spatial mapping attached — the same
form an inline `@kernel(mapping=...)` produces. Applies immediately. Returns
`(kernel_op_ref, call_op_ref)`.

```python
producer, consumer = s.affine(s.loops("i", "j"))
axis = s.compute_at(producer, consumer)
outer, inner = s.split(axis, factor=4)
stage, call = s.outline(inner, func_name="stage0", mapping=[2, 1])
```

## Kernel Composition

### `s.compose()`

`s.compose(*callees, id=None) -> Schedule`

Allo schedules each kernel independently, then stitches them together. When a
top-level kernel invokes a sub-kernel, the compiler specializes a private copy
of the callee named `"{primary}.{callee_primary}"`. `compose` replays a callee's
*entire* schedule onto that copy.

Pass one or more **direct** callees: `s.compose(a, b)` is exactly
`s.compose(a); s.compose(b)`. Each callee must be a kernel `self` calls directly
(a non-direct callee has no copy and raises). `id` selects a specific
specialized/repeat copy when the callee is invoked more than once. Composition is
transitive: a callee that itself composed sub-kernels carries its full include
plan, which is re-prefixed onto this copy. Returns the schedule.

```python
gemm_s = gemm.schedule()
gemm_s.tile(gemm_s.loops("i", "j"), factors=[4, 4])

top_s = top.schedule()         # top invokes gemm
top_s.compose(gemm_s)          # gemm's tiling now applies inside top
top_s.export("vitis").hls_code
```

## Streaming

### `s.streamline()`

`s.streamline(producer, consumer, *, producer_ids=None, consumer_ids=None, lanes=1, depth=2) -> Schedule`

Converts the DRAM memory boundary between two composed stage kernels into an
on-chip stream hand-off (a `to_stream` fusion), so the stages run as a producer/
consumer dataflow pair without round-tripping the intermediate through DRAM.

`producer` and `consumer` are stage kernel names (each a single name or a list):

- **One → one**: every memref the producer only writes and the consumer only
  reads becomes a FIFO; un-convertible boundaries are skipped with a diagnostic.
- **One → many**: the output fans out through a generated `tee` (residual / skip
  connections).
- **Many → one**: inputs fan in through a generated `merge`; each producer must
  fill a disjoint contiguous row-major block.

A `*_ids` list (matching the names) selects specific repeat copies. `lanes`
widens each boundary to `L` parallel FIFOs moving `L` elements per cycle — the
bandwidth lever, valid when the contiguous dimension divides by `L`; `lanes=1`
(default) is a scalar FIFO. `depth` is the FIFO depth (default `2`); on a
reconvergent fork/join the short branch's FIFO must hold the latency skew, or the
dataflow deadlocks — `streamline` warns and names the depth to set. `lanes` and
`depth` must be positive integers. Returns the schedule.

```python
s = top.schedule()
s.compose(stage_a, stage_b, stage_c)
s.streamline("stage_a", "stage_b")               # DRAM -> FIFO
s.streamline("stage_b", "stage_c", lanes=4, depth=8)
s.dataflow()                                     # run the stages concurrently
s.export("vitis").hls_code
```

## Applying Transforms and Ref Lifetime

`apply()` (alias `materialize()`) runs the pending transform script. Tagging
primitives only become visible in `payload` after it runs; structural primitives
apply on their own and need no explicit `apply()`.

When a structural transform invalidates an old ref, recover a live ref one of
three ways: use the refs the primitive returned, select again by name, or call
`s.live(ref)` to rebind a ref whose schedule ID still exists.

```python
i, j = s.loops("i", "j")
outer, inner = s.split(i, factor=4)   # `j` is now from a previous state

j = s.live(j)                         # rebind before reuse
s.pipeline(j, ii=1).apply()
```

A ref consumed by a primitive (e.g. a loop that was split, flattened, or
reordered away) raises `ConsumedHandleError` if reused.

## Exporting to a Backend

`s.export(backend, **kwargs)` applies pending transforms, runs a final cleanup
pass, binds the scheduled module back onto the kernel, and returns a backend
object. `backend` is `"cpu"` or `"vitis"`; keyword arguments are forwarded to the
backend constructor. `export_cpu(**kwargs)` and `export_vitis(**kwargs)` are
shorthands. A schedule with no source kernel (built from a module/string/file)
cannot be exported.

```python
# CPU functional simulation
s.export("cpu")(A, B, C)

# Vitis HLS C++ codegen / csim / synthesis
code = s.export("vitis").hls_code
s.export("vitis", project_path="proj")(A, B, C)          # csim
report = s.export("vitis", part=PART, project_path="proj").synth()
```

See [Simulation](simulation.md) for the backend objects these calls return.

## Debugging

| Helper                                  | Effect                                                        |
| --------------------------------------- | ------------------------------------------------------------- |
| `s.format_tree(*, include_values=True)` | Return a text tree of the current snapshot.                   |
| `s.dump_tree(*, include_values=True)`   | Print and return that tree.                                   |
| `s.dump_transform_script()`             | Return the pending transform script as MLIR text.             |
| `s.debug_dump(*, include_values=True)`  | Print dirty state, op/value counts, the tree, and the script. |
| `s.cleanup_schedule_ids()`              | Remove the internal schedule-ID attributes from the payload.  |

Schedule errors use source-aware diagnostics that point at the Python call site.
The error types live in `allo/schedule/errors.py`:

| Error                          | Raised when                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `ScheduleLookupError`          | A target name or path does not resolve.                                          |
| `AmbiguousLookupError`         | A single-target lookup matches more than one operation.                          |
| `ConsumedHandleError`          | A ref consumed by an earlier transform is reused.                                |
| `ScheduleStateError`           | `payload`/`snapshot` cannot be produced from the pending script.                 |
| `ScheduleTypeError`            | A ref of the wrong kind is passed (e.g. an `OpRef` where a `LoopRef` is needed). |
| `InvalidScheduleArgumentError` | An argument is out of range (e.g. `factor <= 0`).                                |
| `ScheduleTransformError`       | The transform script or resulting payload fails verification.                    |
