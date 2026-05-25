---
title: Allo Scheduling
createdAt: 2026-05-25
summary: User and developer guide for the experimental Allo schedule frontend.
keywords: ["Allo", "Scheduling", "Schedule", "Transform", "MLIR"]
---

%toc%
# Scheduling

This document describes the experimental schedule frontend under
`allo.exp.schedule`. The scheduler works on MLIR modules and builds transform
scripts against stable operation and value references. For normal frontend use,
construct a schedule from a kernel with `kernel.schedule()`.

```python
from allo.exp.lang.core import i32, range
from allo.exp.lang.kernel import kernel


@kernel
def top(A: "i32[16]", B: "i32[16]"):
    for i in range(16, name="i"):
        B[i] = A[i] + 1


s = top.schedule()
i = s.loop("i")

s.pipeline(i, ii=2).unroll(i, factor=4).apply()

print(s.payload)
```

The current scheduler does not expose the older upstream `allo.customize`
scheduling surface. Users select operations, loops, and buffers with typed refs,
apply schedule primitives, and inspect the transformed payload module. The
recommended selection APIs are the short aliases on `Schedule`, such as
`s.loop("i")`, `s.loops("i", "j")`, `s.op("top")`, and `s.buffer("B")`.

## Constructing a Schedule

For frontend kernels, use `Kernel.schedule()`:

```python
s = top.schedule()
```

`Schedule` can also be constructed from an existing MLIR module, a text string,
or a file:

```python
s = Schedule.from_module(module)
s = Schedule.from_string(mlir_text)
s = Schedule.from_file("kernel.mlir")
```

The important public fields are:

| Field | Meaning |
| --- | --- |
| `payload` | The mutable MLIR module being scheduled. |
| `snapshot` | Immutable view of operations and buffer values at the current schedule epoch. |
| `query` | Low-level query object used by the convenience selection aliases. |
| `epoch` | Integer version of the topology snapshot. |
| `dirty` | Whether pending transform operations have not been applied yet. |
| `effects` | Recorded schedule effects for diagnostics and debugging. |

On construction, the scheduler annotates the payload with internal schedule IDs
and collects a snapshot. These IDs are used to reconnect Python references to
payload operations after transforms run.

## Selecting Targets

The user-facing selection methods on `Schedule` return refs directly:

```python
i = s.loop("i")
i, j = s.loops("i", "j")
all_loops = s.loops()

top = s.op("top")
B = s.buffer("B")
```

These methods are aliases over the lower-level `s.query` API:

| Alias | Equivalent query |
| --- | --- |
| `s.op(name, under=None, kind=None, path=None)` | `s.query.op(...).one()` |
| `s.loop(name, under=None, path=None)` | `s.query.loop(...).one()` |
| `s.loops()` | `tuple(s.query.loop().all())` |
| `s.loops("i", "j")` | `s.query.loop().names("i", "j")` |
| `s.buffer(name, under=None, path=None)` | `s.query.buffer(...).one()` |

The lower-level query methods return a `RefSelection`. Use them when you need
`.first()`, `.all()`, `kind=...`, `path=...`, or other MLIR-oriented selection:

| Method | Result |
| --- | --- |
| `query.op(name=None, under=None, kind=None, path=None)` | Select operations. |
| `query.loop(name=None, under=None, path=None)` | Select loop-like operations. |
| `query.loops(...)` | Alias for `query.loop(...)`. |
| `query.buffer(name=None, under=None, path=None)` | Select buffer-like operation operands or results. |

`under` scopes a query or alias to operations nested under another operation ref
or name. `path` selects a specific snapshot path. `kind` matches the MLIR
operation name, for example `affine.for` or `allo.kernel`; it is intentionally
kept on `op`/`query.op` for advanced use.

Loop names come from the frontend iterator names:

```python
@kernel
def top(A: "i32[4,4]", B: "i32[4,4]"):
    for i in range(4, name="i"):
        for j in range(4, name="j"):
            B[i, j] = A[i, j] + 1


s = top.schedule()
i, j = s.loops("i", "j")
```

`grid(..., name="ij")` names the whole loop-like `scf.parallel` operation, not
the individual axes.

Refs are lightweight immutable objects:

| Ref | Meaning |
| --- | --- |
| `OpRef` | Reference to any operation. |
| `LoopRef` | Reference to a loop-like operation. |
| `BufferRef` | Reference to a buffer value owned by an operation. |

Schedule primitives accept refs, names, or iterables of refs/names where a
multi-target operation is meaningful. A name must resolve unambiguously. If a
name is missing or ambiguous, the scheduler reports a source-aware diagnostic.

## Schedule Primitives

Most primitives append one or more transform operations to the pending transform
script and return the schedule for chaining. Call `.apply()` to run pending
transforms unless the primitive documents that it applies immediately.

### Generic Passes

These primitives operate on operation targets. If no target is provided, they
default to the payload root.

| API | Effect |
| --- | --- |
| `schedule.cse(targets=None)` | Apply common subexpression elimination. |
| `schedule.dce(targets=None)` | Apply dead code elimination. |
| `schedule.licm(targets=None)` | Apply loop-invariant code motion. |
| `schedule.canonicalize(targets=None)` | Apply canonicalization patterns. |
| `schedule.apply_patterns(patterns, targets=None)` | Apply named rewrite patterns. Currently supports `"canonicalize"`. |

Example:

```python
schedule.cse().dce().canonicalize().apply()
```

### Loop Tags and Memory Tags

These primitives attach schedule attributes without changing IR topology, so
existing refs stay live after `.apply()`.

| API | Effect |
| --- | --- |
| `schedule.pipeline(targets=None, ii=1)` | Mark loop targets with a pipeline initiation interval. |
| `schedule.unroll(targets=None, factor=0, tag_only=True)` | Mark loop targets for unrolling. `factor=0` means full unroll. |
| `schedule.partition(targets, dim=0, kind=Schedule.Complete, factor=0)` | Attach an Allo partition attribute to buffer targets. |

Partition kind is one of `Schedule.Complete`, `Schedule.Block`, or
`Schedule.Cyclic`. Complete partition uses `factor=0`; block and cyclic
partitions require a positive factor.

```python
loop = s.loop("i")
A = s.buffer("A")

s.pipeline(loop, ii=2)
s.partition(A, dim=1, kind=Schedule.Cyclic, factor=4)
s.apply()
```

`unroll(..., tag_only=False)` performs physical unrolling and applies
immediately because it changes IR topology.

### Loop Restructuring

These primitives change the loop nest. They apply immediately and return live
refs for the new topology.

| API | Effect |
| --- | --- |
| `schedule.split(target, factor=1)` | Split one loop and return `(outer, inner)`. |
| `schedule.reorder(targets)` | Reorder affine loops and return refs in the requested order. |
| `schedule.tile(targets, factors=1)` | Tile a loop nest and return `(tiles, points)`. |
| `schedule.flatten(targets)` | Flatten two or more loops and return the new loop ref. |

```python
i, j = s.loops("i", "j")

outer, inner = s.split(i, factor=4)
s.pipeline(inner, ii=1).apply()

tiles, points = s.tile([outer, j], factors=[2, 4])
s.pipeline(points[-1], ii=1).apply()
```

`reorder`, `tile`, and `flatten` expect affine loops. `tile` accepts either a
single integer factor, broadcast to all target loops, or an iterable with one
factor per loop.

### Data Movement and Outlining

These primitives move computation, buffers, or regions. Primitives that return
new refs apply immediately.

| API | Effect |
| --- | --- |
| `schedule.affine(targets=None)` | Raise loop targets to affine form and return live loop refs. |
| `schedule.compute_at(target, axis)` | Move a producer operation to the given affine loop axis and return the live axis ref. |
| `schedule.buffer_at(target, axis)` | Create a localized buffer at an affine loop axis and return the new buffer ref. |
| `schedule.outline(target, func_name, mapping=None)` | Outline an operation into a new function or Allo kernel and return `(kernel, call)`. |

`outline` emits a normal `func.func`/`call` pair when `mapping` is `None`. When
`mapping` is an integer or a sequence of positive integers, it emits an
`allo.kernel` and `allo.invoke` with the mapping attached.

```python
producer_loop, consumer_loop = s.affine(s.loops("i", "j"))

axis = s.compute_at(producer_loop, consumer_loop)
outer, inner = s.split(axis, factor=4)

kernel, call = s.outline(inner, func_name="stage0", mapping=[2, 1])
```

## Applying Transforms

`apply()` verifies and runs the pending transform script against `payload`.

```python
s.pipeline(loop, ii=2)
s.unroll(loop, factor=4)
s.apply()
```

The scheduler distinguishes topology-changing effects from attribute-only
effects:

- Attribute-only effects such as `pipeline`, tag-only `unroll`, and `partition`
  do not bump `epoch`; existing refs remain live.
- Topology-changing effects rebuild the snapshot and increment `epoch`.
  Existing refs from older epochs become stale.
- Primitives that must return newly created refs, such as `split`, `tile`,
  `flatten`, `outline`, `compute_at`, and `buffer_at`, call `apply()`
  internally.

When a topology-changing transform invalidates an old ref, use the refs returned
by the primitive, select again with an alias such as `s.loop("j")`, or call
`s.live(ref)` to rebind a ref whose schedule ID still exists.

```python
i, j = s.loops("i", "j")
outer, inner = s.split(i, factor=4)

# `j` came from the previous epoch. Rebind it before using it.
j = s.live(j)
s.pipeline(j, ii=1).apply()
```

## Scheduler Model

The scheduler has three layers:

1. The payload module, which is the MLIR module being transformed.
2. An immutable snapshot of payload operations and buffer values, indexed by
   schedule ID, name, kind, and path.
3. A transform script module that contains MLIR transform dialect operations.

Queries read only from the snapshot, and the `s.loop`/`s.op`/`s.buffer` aliases
are thin wrappers around those queries. Schedule primitives resolve refs against
the current snapshot, append transform operations to the transform script, and
record an `Effect`. `apply()` verifies the transform script, applies it to the
payload, verifies the payload, refreshes the snapshot, and starts a fresh
transform script for the next batch.

This model makes scheduling refs explicit. A `LoopRef` or `BufferRef` is valid
only for the epoch in which it was created. If a transform only annotates the IR,
the epoch stays the same. If a transform changes topology, the epoch advances
and the scheduler requires callers to use new refs.

## Debugging

The scheduler exposes a few inspection helpers:

| API | Effect |
| --- | --- |
| `schedule.format_tree(include_values=True)` | Return a text tree of the current snapshot. |
| `schedule.dump_tree(include_values=True)` | Print and return the current snapshot tree. |
| `schedule.dump_transform_script()` | Return the pending transform script as MLIR text. |
| `schedule.debug_dump(include_values=True)` | Print epoch, dirty state, snapshot size, effects, and tree. |
| `schedule.cleanup_schedule_ids()` | Remove internal schedule ID attributes from the payload. |

Schedule errors use source-aware diagnostics when the Python call site is
available. Lookup errors report missing or ambiguous targets. Stale ref errors
include the ref epoch, current epoch, and the last topology-changing transform
when available.
