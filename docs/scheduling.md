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
scripts against stable operation and value references. The public entry point is
`Schedule`.

```python
from allo.exp.schedule import Schedule


schedule = Schedule.from_string(mlir_text)
loop = schedule.query.loop().one()

schedule.pipeline(loop, ii=2).unroll(loop, factor=4).apply()

print(schedule.payload)
```

The current scheduler is intentionally close to MLIR. It does not yet expose the
older upstream `allo.customize` scheduling surface. Users select operations,
loops, and buffers from the payload IR, apply schedule primitives, and inspect
the transformed payload module.

## Constructing a Schedule

`Schedule` can be constructed from an existing MLIR module, a text string, or a
file:

```python
schedule = Schedule.from_module(module)
schedule = Schedule.from_string(mlir_text)
schedule = Schedule.from_file("kernel.mlir")
```

The important public fields are:

| Field | Meaning |
| --- | --- |
| `payload` | The mutable MLIR module being scheduled. |
| `snapshot` | Immutable view of operations and buffer values at the current schedule epoch. |
| `query` | Query object used to select operations, loops, and buffers. |
| `epoch` | Integer version of the topology snapshot. |
| `dirty` | Whether pending transform operations have not been applied yet. |
| `effects` | Recorded schedule effects for diagnostics and debugging. |

On construction, the scheduler annotates the payload with internal schedule IDs
and collects a snapshot. These IDs are used to reconnect Python references to
payload operations after transforms run.

## Querying Targets

Queries return a `RefSelection`. A selection can be converted to refs with
`.one()`, `.first()`, `.all()`, or `.names(...)`.

```python
loop = schedule.query.loop().one()
loops = schedule.query.loop().all()
i, j = schedule.query.loop().names("i", "j")

kernel = schedule.query.op("kernel").one()
stores = schedule.query.op(kind="affine.store").all()
buffer = schedule.query.buffer("tmp").one()
```

The query methods are:

| Method | Result |
| --- | --- |
| `query.op(name=None, under=None, kind=None, path=None)` | Select operations. |
| `query.loop(name=None, under=None, path=None)` | Select loop-like operations. |
| `query.loops(...)` | Alias for `query.loop(...)`. |
| `query.buffer(name=None, under=None, path=None)` | Select buffer-like operation operands or results. |

`under` scopes a query to operations nested under another operation ref or name.
`path` selects a specific snapshot path. `kind` matches the MLIR operation name,
for example `affine.for` or `func.func`.

Refs are lightweight immutable objects:

| Ref | Meaning |
| --- | --- |
| `OpRef` | Reference to any operation. |
| `LoopRef` | Reference to a loop-like operation. |
| `BufferRef` | Reference to a buffer value owned by an operation. |

Schedule primitives accept refs, names, or iterables of refs/names where a
multi-target operation is meaningful. A name must resolve unambiguously.

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
loop = schedule.query.loop().one()
buf = schedule.query.buffer("A").one()

schedule.pipeline(loop, ii=2)
schedule.partition(buf, dim=1, kind=Schedule.Cyclic, factor=4)
schedule.apply()
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
i, j = schedule.query.loop().all()

outer, inner = schedule.split(i, factor=4)
schedule.pipeline(inner, ii=1).apply()

tiles, points = schedule.tile([outer, j], factors=[2, 4])
schedule.pipeline(points[-1], ii=1).apply()
```

`reorder`, `tile`, and `flatten` expect affine loops. `tile` accepts either a
single integer factor, broadcast to all target loops, or an iterable with one
factor per loop.

### Data Movement and Outlining

These primitives move computation, buffers, or regions. Primitives that return
new refs apply immediately; `polyhedral` is queued like other target transforms
and should be followed by `.apply()` unless it is part of a larger pending
chain.

| API | Effect |
| --- | --- |
| `schedule.polyhedral(targets=None)` | Raise loop targets to affine form. This queues a transform; call `.apply()` unless it is chained with other pending transforms. |
| `schedule.compute_at(target, axis)` | Move a producer operation to the given affine loop axis and return the live axis ref. |
| `schedule.buffer_at(target, axis)` | Create a localized buffer at an affine loop axis and return the new buffer ref. |
| `schedule.outline(target, func_name, mapping=None)` | Outline an operation into a new function or Allo kernel and return `(kernel, call)`. |

`outline` emits a normal `func.func`/`call` pair when `mapping` is `None`. When
`mapping` is an integer or a sequence of positive integers, it emits an
`allo.kernel` and `allo.invoke` with the mapping attached.

```python
producer = schedule.query.op("producer_store").one()
axis = schedule.query.loop("consumer_loop").one()

axis = schedule.compute_at(producer, axis)
outer, inner = schedule.split(axis, factor=4)

kernel, call = schedule.outline(inner, func_name="stage0", mapping=[2, 1])
```

## Applying Transforms

`apply()` verifies and runs the pending transform script against `payload`.

```python
schedule.pipeline(loop, ii=2)
schedule.unroll(loop, factor=4)
schedule.apply()
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
by the primitive, run a fresh query, or call `schedule.live(ref)` to rebind a ref
whose schedule ID still exists.

```python
i, j = schedule.query.loop().all()
outer, inner = schedule.split(i, factor=4)

# `j` came from the previous epoch. Rebind it before using it.
j = schedule.live(j)
schedule.pipeline(j, ii=1).apply()
```

## Scheduler Model

The scheduler has three layers:

1. The payload module, which is the MLIR module being transformed.
2. An immutable snapshot of payload operations and buffer values, indexed by
   schedule ID, name, kind, and path.
3. A transform script module that contains MLIR transform dialect operations.

Queries read only from the snapshot. Schedule primitives resolve refs against
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
