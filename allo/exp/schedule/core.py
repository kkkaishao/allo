from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from typing import Any

from .errors import (
    InvalidScheduleArgumentError,
    ScheduleLookupError,
    ScheduleTypeError,
    ScheduleTransformError,
)
from .model import (
    BufferRef,
    Effect,
    LoopRef,
    OpRef,
    Ref,
    ScheduleSnapshot,
    SingleTarget,
    Targets,
)
from .query import Query
from .script import TransformScript
from ..._mlir import ir
from ..._mlir import schedule as schedule_d
from ..._mlir.schedule import ScheduleOpTrait
from ..._mlir.dialects import allo as allo_d
from ..._mlir.dialects import transform as t
from ..._mlir.dialects.transform import allo as ta
from ..._mlir.dialects.transform import interpreter
from ..logging import log_debug, text_tail


def _within_context(method):
    """Run a schedule primitive under ``with self.context`` so upstream ODS attr
    builders (StrArrayAttr/I64Attr/…) can resolve the MLIR context."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.context:
            return method(self, *args, **kwargs)

    return wrapper


class Schedule:
    """Experimental schedule frontend backed by immutable IR snapshots."""

    Complete = 0
    Block = 1
    Cyclic = 2

    payload: ir.Module
    snapshot: ScheduleSnapshot
    query: Query
    script: TransformScript

    def __init__(self, module: ir.Module, context: ir.Context | None = None):
        self.payload = module
        self.context = context if context is not None else module.context
        allo_d.register_extensions(self.context)
        self.epoch = 0
        self.dirty = False
        self.effects: list[Effect] = []
        self._pending_effects: list[Effect] = []
        self._effect_counter = 0
        self._generated_ids: set[str] = set()
        self.last_effect: str | None = None
        schedule_d.annotate_schedule_ids(self.payload)
        self.snapshot = self._collect_snapshot()
        self.query = Query(self)
        self.script = TransformScript(self)

    @classmethod
    def from_module(
        cls, module: ir.Module, context: ir.Context | None = None
    ) -> Schedule:
        return cls(module, context)

    @classmethod
    def from_string(cls, text: str) -> Schedule:
        context = ir.Context()
        allo_d.register_dialect(context)
        module = ir.Module.parse(text, context)
        return cls(module, context)

    @classmethod
    def from_file(cls, path: str) -> Schedule:
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_string(handle.read())

    def cleanup_schedule_ids(self) -> Schedule:
        schedule_d.cleanup_schedule_ids(self.payload)
        log_debug("removed schedule ids from payload IR")
        return self

    def live(self, ref: Ref) -> Ref:
        if not isinstance(ref, Ref):
            raise InvalidScheduleArgumentError(
                f"live expects a schedule ref, got {type(ref).__name__}"
            )
        if isinstance(ref, BufferRef):
            if ref.id not in self.snapshot.values_by_id:
                raise ScheduleLookupError(f"buffer ref id '{ref.id}' is no longer live")
            return self.snapshot.buffer_ref(ref.id)
        if ref.id not in self.snapshot.ops_by_id:
            raise ScheduleLookupError(f"operation ref id '{ref.id}' is no longer live")
        if isinstance(ref, LoopRef):
            return self.snapshot.loop_ref(ref.id)
        assert isinstance(ref, OpRef), f"unsupported ref type: {type(ref)}"
        return self.snapshot.op_ref(ref.id)

    #####################################
    # Alias methods for query operations
    # To simplify the use of the schedule API
    ####################################

    def op(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        kind: str | None = None,
        path: str | None = None,
    ) -> OpRef:
        return self.query.op(name, under=under, kind=kind, path=path).one()

    def loop(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> LoopRef:
        return self.query.loop(name, under=under, path=path).one()

    def loops(
        self,
        *names: str,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> tuple[LoopRef, ...]:
        selection = self.query.loop(under=under, path=path)
        if names:
            return selection.names(*names)
        return tuple(selection.all())

    def buffer(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> BufferRef:
        return self.query.buffer(name, under=under, path=path).one()

    @_within_context
    def cse(self, targets: Targets = None) -> Schedule:
        ops = self._resolve_op_targets(targets, "cse")
        self.script.set_callsite_loc()
        for op in ops:
            t.ApplyCommonSubexpressionEliminationOp(
                self.script.op_handle(op), **self.script.kw
            )

        self._mark_dirty()
        self._record_effect(
            "cse",
            topology=True,
            targets=[op.path for op in ops],
        )
        return self

    @_within_context
    def dce(self, targets: Targets = None) -> Schedule:
        ops = self._resolve_op_targets(targets, "dce")
        self.script.set_callsite_loc()
        for op in ops:
            t.ApplyDeadCodeEliminationOp(self.script.op_handle(op), **self.script.kw)

        self._mark_dirty()
        self._record_effect(
            "dce",
            topology=True,
            targets=[op.path for op in ops],
        )
        return self

    @_within_context
    def licm(self, targets: Targets = None) -> Schedule:
        ops = self._resolve_op_targets(targets, "licm")
        self.script.set_callsite_loc()
        for op in ops:
            t.ApplyLoopInvariantCodeMotionOp(
                self.script.op_handle(op), **self.script.kw
            )

        self._mark_dirty()
        self._record_effect(
            "licm",
            topology=True,
            targets=[op.path for op in ops],
        )
        return self

    @_within_context
    def apply_patterns(
        self,
        patterns: str | Iterable[str],
        targets: Targets = None,
    ) -> Schedule:
        pattern_names = [patterns] if isinstance(patterns, str) else list(patterns)
        if not pattern_names:
            raise InvalidScheduleArgumentError("apply_patterns requires a pattern")

        supported_patterns = {"canonicalize": t.ApplyCanonicalizationPatternsOp}
        pattern_ops = []
        for pattern in pattern_names:
            op = supported_patterns.get(pattern)
            if op is None:
                raise InvalidScheduleArgumentError(
                    f"unsupported pattern '{pattern}' in apply_patterns"
                )
            pattern_ops.append(op)

        ops = self._resolve_op_targets(targets, "apply_patterns")
        self.script.set_callsite_loc()
        for op in ops:
            apply_op = t.ApplyPatternsOp(self.script.op_handle(op), **self.script.kw)
            region = apply_op.regions[0]
            body = region.blocks[0] if len(region.blocks) else region.blocks.append()
            ip = self.script.builder.save_insertion_point()
            self.script.builder.set_insertion_point_to_end(body)
            for pattern_op in pattern_ops:
                pattern_op(**self.script.kw)
            self.script.builder.restore_insertion_point(ip)

        self._mark_dirty()
        self._record_effect(
            "apply_patterns",
            topology=True,
            targets=[op.path for op in ops],
            patterns=pattern_names,
        )
        return self

    def canonicalize(self, targets: Targets = None) -> Schedule:
        return self.apply_patterns("canonicalize", targets)

    @_within_context
    def pipeline(self, targets: Targets = None, *, ii: int = 1) -> Schedule:
        self._require_int("pipeline ii", ii)
        if ii <= 0:
            raise InvalidScheduleArgumentError(
                f"pipeline ii must be positive, got {ii}"
            )

        loops = self._resolve_loop_targets(targets, "pipeline")
        self.script.set_callsite_loc()
        for loop in loops:
            ta.TagPipelineOp(self.script.op_handle(loop), ii, **self.script.kw)

        self._mark_dirty()
        self._record_effect(
            "pipeline",
            topology=False,
            targets=[loop.path for loop in loops],
            ii=ii,
        )
        return self

    @_within_context
    def unroll(
        self,
        targets: Targets = None,
        *,
        factor: int = 0,
        tag_only: bool = True,
    ) -> Schedule:
        self._require_int("unroll factor", factor)
        if factor < 0:
            raise InvalidScheduleArgumentError(
                f"unroll factor must be non-negative, got {factor}"
            )

        loops = self._resolve_loop_targets(targets, "unroll")
        self.script.set_callsite_loc()
        for loop in loops:
            ta.AlloLoopUnrollOp(
                self.script.op_handle(loop),
                factor,
                tag_only=tag_only,
                **self.script.kw,
            )

        self._mark_dirty()
        self._record_effect(
            "unroll",
            topology=not tag_only,
            targets=[loop.path for loop in loops],
            factor=factor,
            tag_only=tag_only,
        )
        if not tag_only:
            self.apply()
        return self

    @_within_context
    def partition(
        self,
        targets: Targets,
        *,
        dim: int = 0,
        kind=Complete,
        factor: int = 0,
    ) -> Schedule:
        self._require_int("partition dim", dim)
        self._require_int("partition factor", factor)
        if dim < 0:
            raise InvalidScheduleArgumentError(
                f"partition dim must be non-negative, got {dim}"
            )
        if kind not in (self.Complete, self.Block, self.Cyclic):
            raise InvalidScheduleArgumentError(
                "partition kind must be Schedule.Complete, Schedule.Block, "
                "or Schedule.Cyclic"
            )
        if kind == self.Complete:
            if factor != 0:
                raise InvalidScheduleArgumentError(
                    "complete partition cannot have non-zero factor"
                )
        elif factor <= 0:
            raise InvalidScheduleArgumentError(
                f"{self._partition_kind_name(kind)} partition factor must be "
                f"positive, got {factor}"
            )

        buffers = self._resolve_buffer_targets(targets, "partition")
        axis = allo_d.PartitionAxisAttr.get(kind, factor, dim, self.context)
        part = allo_d.PartitionAttr.get([axis], self.context)
        self.script.set_callsite_loc()
        for buffer in buffers:
            ta.PartitionOp(self.script.value_handle(buffer), part, **self.script.kw)

        self._mark_dirty()
        self._record_effect(
            "partition",
            topology=False,
            targets=[buffer.path for buffer in buffers],
            dim=dim,
            kind=self._partition_kind_name(kind),
            factor=factor,
        )
        return self

    @_within_context
    def affine(self, targets: Targets = None) -> list[LoopRef]:
        loops = self._resolve_loop_targets(targets, "affine")
        self.script.set_callsite_loc()
        for loop in loops:
            raised = ta.RaiseToAffineOp(
                self.script.any_op_type, self.script.op_handle(loop), **self.script.kw
            ).result
            self.script.annotate_schedule_id(raised, loop.id)
            if loop.name is not None:
                self.script.annotate_schedule_name(raised, loop.name)
            self.script.set_op_handle(loop, raised)

        self._mark_dirty()
        self._record_effect(
            "affine",
            topology=True,
            targets=[loop.path for loop in loops],
        )
        self.apply()
        return [self.snapshot.loop_ref(loop.id) for loop in loops]

    @_within_context
    def compute_at(self, target: SingleTarget, axis: SingleTarget) -> LoopRef:
        producer = self._resolve_single_op_target(target, "compute_at target")
        axis_loop = self._require_affine_for(
            self._resolve_single_loop_target(axis, "compute_at axis"),
            "compute_at axis",
        )

        self.script.set_callsite_loc()
        ta.ComputeAtOp(
            self.script.op_handle(producer),
            self.script.op_handle(axis_loop),
            **self.script.kw,
        )

        self._mark_dirty()
        self._record_effect(
            "compute_at",
            topology=True,
            target=producer.path,
            axis=axis_loop.path,
        )
        self.apply()
        return self.snapshot.loop_ref(axis_loop.id)

    @_within_context
    def buffer_at(self, target: SingleTarget, axis: SingleTarget) -> BufferRef:
        buffer = self._resolve_single_buffer_target(target, "buffer_at target")
        axis_loop = self._require_affine_for(
            self._resolve_single_loop_target(axis, "buffer_at axis"),
            "buffer_at axis",
        )
        local_id = self._fresh_schedule_id(f"{buffer.id}.local")

        self.script.set_callsite_loc()
        local = ta.BufferAtOp(
            self.script.any_value_type,
            self.script.value_handle(buffer),
            self.script.op_handle(axis_loop),
            **self.script.kw,
        ).result
        local_alloc = self.script.defining_op_handle(local)
        self.script.annotate_schedule_id(local_alloc, local_id)

        self._mark_dirty()
        self._record_effect(
            "buffer_at",
            topology=True,
            target=buffer.path,
            axis=axis_loop.path,
            result=local_id,
        )
        self.apply()
        return self.snapshot.buffer_ref(f"{local_id}:res0")

    @_within_context
    def outline(
        self,
        target: SingleTarget,
        *,
        func_name: str,
        mapping: Sequence[int] | int | None = None,
    ) -> tuple[OpRef, OpRef]:
        if not isinstance(func_name, str) or not func_name:
            raise InvalidScheduleArgumentError("outline requires a non-empty func_name")
        source = self._resolve_single_op_target(target, "outline target")
        if source.id == self.snapshot.root_id:
            raise InvalidScheduleArgumentError("outline cannot target the payload root")
        mapping_values = self._normalize_mapping(mapping, "outline mapping")
        kernel_id = self._fresh_schedule_id(func_name)
        call_id = self._fresh_schedule_id(f"{source.id}.call")

        self.script.set_callsite_loc()
        any_op = self.script.any_op_type
        if mapping_values is None:
            outlined = ta.OutlineOp(
                any_op,
                any_op,
                self.script.op_handle(source),
                func_name,
                **self.script.kw,
            )
        else:
            outlined = ta.OutlineOp(
                any_op,
                any_op,
                self.script.op_handle(source),
                func_name,
                mapping=mapping_values,
                **self.script.kw,
            )
        self.script.annotate_schedule_id(outlined.results[0], kernel_id)
        self.script.annotate_schedule_id(outlined.results[1], call_id)

        effect_data = {
            "target": source.path,
            "func_name": func_name,
            "kernel": kernel_id,
            "call": call_id,
        }
        if mapping_values is not None:
            effect_data["mapping"] = mapping_values
        self._mark_dirty()
        self._record_effect(
            "outline",
            topology=True,
            **effect_data,
        )
        self.apply()
        return self.snapshot.op_ref(kernel_id), self.snapshot.op_ref(call_id)

    @_within_context
    def split(
        self,
        target: SingleTarget | None = None,
        *,
        factor: int = 1,
    ) -> tuple[LoopRef, LoopRef]:
        self._require_int("split factor", factor)
        if factor <= 0:
            raise InvalidScheduleArgumentError(
                f"split factor must be positive, got {factor}"
            )

        loops = self._resolve_loop_targets(target, "split")
        if len(loops) != 1:
            raise InvalidScheduleArgumentError("split requires exactly one loop target")
        loop = loops[0]
        outer_id = self._fresh_schedule_id(f"{loop.id}.outer")
        inner_id = self._fresh_schedule_id(f"{loop.id}.inner")

        self.script.set_callsite_loc()
        any_op = self.script.any_op_type
        split_op = ta.LoopSplitOp(
            any_op, any_op, self.script.op_handle(loop), factor, **self.script.kw
        )
        outer = split_op.results[0]
        inner = split_op.results[1]
        self.script.annotate_schedule_id(outer, outer_id)
        self.script.annotate_schedule_id(inner, inner_id)

        self._mark_dirty()
        self._record_effect(
            "split",
            topology=True,
            target=loop.path,
            factor=factor,
            outer=outer_id,
            inner=inner_id,
        )
        self.apply()
        return self.snapshot.loop_ref(outer_id), self.snapshot.loop_ref(inner_id)

    @_within_context
    def reorder(self, targets: Targets) -> tuple[LoopRef, ...]:
        desired = [
            self._require_affine_for(loop, "reorder target")
            for loop in self._resolve_loop_targets(targets, "reorder")
        ]
        if len(desired) < 2:
            raise InvalidScheduleArgumentError("reorder requires at least two loops")
        desired_ids = [loop.id for loop in desired]
        if len(set(desired_ids)) != len(desired_ids):
            raise InvalidScheduleArgumentError("reorder targets must be unique")

        current = sorted(desired, key=lambda loop: self.snapshot.depth(loop.id))
        current_ids = [loop.id for loop in current]
        permutation = [current_ids.index(loop.id) for loop in desired]

        self.script.set_callsite_loc()
        handles = [self.script.op_handle(loop) for loop in current]
        merged = t.MergeHandlesOp(handles, deduplicate=False, **self.script.kw).result
        ta.LoopReorderOp(merged, permutation, **self.script.kw)

        self._mark_dirty()
        self._record_effect(
            "reorder",
            topology=True,
            targets=[loop.path for loop in current],
            order=[loop.path for loop in desired],
        )
        self.apply()
        return tuple(self.snapshot.loop_ref(loop_id) for loop_id in desired_ids)

    @_within_context
    def tile(
        self,
        targets: Targets = None,
        *,
        factors: int | Iterable[int] = 1,
    ) -> tuple[list[LoopRef], list[LoopRef]]:
        loops = self._resolve_loop_targets(targets, "tile")
        factor_list = self._normalize_tile_factors(factors, len(loops))
        depth_ordered = sorted(loops, key=lambda loop: self.snapshot.depth(loop.id))
        tile_ids = [
            self._fresh_schedule_id(f"{loop.id}.tile") for loop in depth_ordered
        ]
        point_ids = [
            self._fresh_schedule_id(f"{loop.id}.point") for loop in depth_ordered
        ]

        self.script.set_callsite_loc()
        handles = [self.script.op_handle(loop) for loop in loops]
        merged = t.MergeHandlesOp(handles, deduplicate=True, **self.script.kw).result
        any_op = self.script.any_op_type
        tiled = ta.LoopTileOp(any_op, any_op, merged, factor_list, **self.script.kw)
        self._annotate_split_results(tiled.results[0], tile_ids)
        self._annotate_split_results(tiled.results[1], point_ids)

        self._mark_dirty()
        self._record_effect(
            "tile",
            topology=True,
            targets=[loop.path for loop in loops],
            factors=factor_list,
            tiles=tile_ids,
            points=point_ids,
        )
        self.apply()
        return (
            [self.snapshot.loop_ref(schedule_id) for schedule_id in tile_ids],
            [self.snapshot.loop_ref(schedule_id) for schedule_id in point_ids],
        )

    @_within_context
    def flatten(self, targets: Targets) -> LoopRef:
        loops = self._resolve_loop_targets(targets, "flatten")
        if len(loops) < 2:
            raise InvalidScheduleArgumentError(
                "flatten requires at least two loop targets"
            )
        outermost = min(loops, key=lambda loop: self.snapshot.depth(loop.id))
        flat_id = self._fresh_schedule_id(f"{outermost.id}.flat")

        self.script.set_callsite_loc()
        handles = [self.script.op_handle(loop) for loop in loops]
        merged = t.MergeHandlesOp(handles, deduplicate=True, **self.script.kw).result
        flattened = ta.LoopFlattenOp(
            self.script.any_op_type, merged, **self.script.kw
        ).result
        self.script.annotate_schedule_id(flattened, flat_id)

        self._mark_dirty()
        self._record_effect(
            "flatten",
            topology=True,
            targets=[loop.path for loop in loops],
            result=flat_id,
        )
        self.apply()
        return self.snapshot.loop_ref(flat_id)

    def apply(self) -> Schedule:
        if not self.dirty:
            return self
        assert self._pending_effects, "dirty schedule has no pending effects"
        topology_changed = any(effect.topology for effect in self._pending_effects)

        if not self.script.module.operation.verify():
            raise ScheduleTransformError(
                "transform script verification failed",
                notes=self._transform_error_notes(),
            )

        try:
            interpreter.apply_named_sequence(
                self.payload.operation,
                self.script.sequence.operation,
                self.script.module.operation,
            )
        except Exception as exc:  # interpreter raises (no failed/err tuple)
            raise ScheduleTransformError(
                "failed to apply transform script",
                notes=self._transform_error_notes(str(exc)),
            ) from exc
        if not self.payload.operation.verify():
            raise ScheduleTransformError(
                "payload module verification failed after scheduling",
                notes=self._transform_error_notes(str(self.payload)),
            )

        self.refresh_snapshot(
            effect=self._last_pending_topology_effect_id(),
            topology=topology_changed,
        )
        self.script = TransformScript(self)
        self.dirty = False
        self._pending_effects.clear()
        return self

    def refresh_snapshot(
        self,
        *,
        effect: str | None = None,
        topology: bool = True,
    ) -> Schedule:
        if topology:
            self.epoch += 1
            self.last_effect = effect
        else:
            assert effect is None, "non-topology refresh cannot set last effect"
        schedule_d.annotate_schedule_ids(self.payload)
        self.snapshot = self._collect_snapshot()
        self.query = Query(self)
        return self

    def format_tree(self, *, include_values: bool = True) -> str:
        return self.snapshot.format_tree(include_values=include_values)

    def dump_tree(self, *, include_values: bool = True) -> str:
        text = self.format_tree(include_values=include_values)
        print(text)
        return text

    def dump_transform_script(self) -> str:
        return str(self.script.module)

    def debug_dump(self, *, include_values: bool = True) -> Schedule:
        print("=== Schedule State ===")
        print(f"epoch={self.epoch}")
        print(f"dirty={self.dirty}")
        print(f"ops={len(self.snapshot.ops)}")
        print(f"values={len(self.snapshot.values)}")
        print(f"effects={len(self.effects)}")
        print("--- tree ---")
        print(self.format_tree(include_values=include_values))
        if self.dirty:
            print("--- transform_script ---")
            print(self.dump_transform_script())
        return self

    def _collect_snapshot(self) -> ScheduleSnapshot:
        raw = schedule_d.collect_schedule_snapshot(self.payload)
        return ScheduleSnapshot.from_raw(raw, epoch=self.epoch)

    def _resolve_op_targets(self, targets: Targets, desc: str) -> list[OpRef]:
        if targets is None:
            return [self.snapshot.op_ref(self.snapshot.root_id)]
        return [
            self.query.resolve_op(target, desc=desc)
            for target in self._targets(targets, desc)
        ]

    def _resolve_loop_targets(self, targets: Targets, desc: str) -> list[LoopRef]:
        if targets is None:
            return [self.query.loop().one()]
        return [
            self.query.resolve_loop(target) for target in self._targets(targets, desc)
        ]

    def _resolve_buffer_targets(self, targets: Targets, desc: str) -> list[BufferRef]:
        if targets is None:
            raise InvalidScheduleArgumentError(f"{desc} requires a buffer target")
        return [
            self.query.resolve_buffer(target) for target in self._targets(targets, desc)
        ]

    def _resolve_single_op_target(
        self, target: SingleTarget | None, desc: str
    ) -> OpRef:
        if target is None:
            raise InvalidScheduleArgumentError(f"{desc} requires a target")
        ops = self._resolve_op_targets(target, desc)
        if len(ops) != 1:
            raise InvalidScheduleArgumentError(f"{desc} requires exactly one target")
        return ops[0]

    def _resolve_single_loop_target(
        self, target: SingleTarget | None, desc: str
    ) -> LoopRef:
        if target is None:
            raise InvalidScheduleArgumentError(f"{desc} requires a target")
        loops = self._resolve_loop_targets(target, desc)
        if len(loops) != 1:
            raise InvalidScheduleArgumentError(f"{desc} requires exactly one loop")
        return loops[0]

    def _resolve_single_buffer_target(
        self, target: SingleTarget | None, desc: str
    ) -> BufferRef:
        if target is None:
            raise InvalidScheduleArgumentError(f"{desc} requires a target")
        buffers = self._resolve_buffer_targets(target, desc)
        if len(buffers) != 1:
            raise InvalidScheduleArgumentError(f"{desc} requires exactly one buffer")
        return buffers[0]

    def _require_affine_for(self, loop: LoopRef, desc: str) -> LoopRef:
        node = self.snapshot.ops_by_id[loop.id]
        if not node.has_trait(ScheduleOpTrait.AFFINE_FOR):
            raise ScheduleTypeError(
                f"{desc} must be an affine.for loop, got {loop.describe()} "
                f"with kind '{loop.kind}'"
            )
        return loop

    def _targets(self, targets: Targets, desc: str) -> list[SingleTarget]:
        if isinstance(targets, (Ref, str)):
            return [targets]
        if not isinstance(targets, Iterable):
            raise InvalidScheduleArgumentError(
                f"{desc} target must be a ref, name, or iterable of refs/names, "
                f"got {type(targets).__name__}"
            )
        out = list(targets)
        if not out:
            raise InvalidScheduleArgumentError(f"{desc} requires at least one target")
        for target in out:
            if not isinstance(target, (Ref, str)):
                raise InvalidScheduleArgumentError(
                    f"{desc} target must be a ref or name, got "
                    f"{type(target).__name__}"
                )
        return out

    def _require_int(self, desc: str, value: int) -> None:
        if type(value) is not int:
            raise InvalidScheduleArgumentError(
                f"{desc} must be an int, got {type(value).__name__}"
            )

    def _mark_dirty(self) -> None:
        self.dirty = True

    def _record_effect(
        self,
        op_name: str,
        *,
        topology: bool,
        **data: Any,
    ) -> Effect:
        self._effect_counter += 1
        effect_id = f"{op_name}:{self._effect_counter}"
        effect = Effect(
            id=effect_id,
            name=op_name,
            epoch=self.epoch,
            topology=topology,
            data=data,
        )
        self.effects.append(effect)
        self._pending_effects.append(effect)
        return effect

    def _last_pending_topology_effect_id(self) -> str | None:
        for effect in reversed(self._pending_effects):
            if effect.topology:
                return effect.id
        return None

    def _fresh_schedule_id(self, base: str) -> str:
        used = set(self.snapshot.ops_by_id) | self._generated_ids
        candidate = base
        suffix = 0
        while candidate in used:
            suffix += 1
            candidate = f"{base}.{suffix}"
        self._generated_ids.add(candidate)
        return candidate

    def _normalize_tile_factors(
        self,
        factors: int | Iterable[int],
        expected: int,
    ) -> list[int]:
        if type(factors) is int:
            out = [factors] * expected
        elif isinstance(factors, Iterable):
            out = list(factors)
        else:
            raise InvalidScheduleArgumentError(
                f"tile factors must be an int or iterable of ints, got "
                f"{type(factors).__name__}"
            )
        if len(out) != expected:
            raise InvalidScheduleArgumentError(
                f"tile expects {expected} factors, got {len(out)}"
            )
        for factor in out:
            self._require_int("tile factor", factor)
            if factor <= 0:
                raise InvalidScheduleArgumentError(
                    f"tile factors must be positive, got {factor}"
                )
        return out

    def _normalize_mapping(
        self,
        mapping: Sequence[int] | int | None,
        desc: str,
    ) -> list[int] | None:
        if mapping is None:
            return None
        if type(mapping) is int:
            out = [mapping]
        elif isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
            out = list(mapping)
        else:
            raise InvalidScheduleArgumentError(
                f"{desc} must be an int, sequence of ints, or None, got "
                f"{type(mapping).__name__}"
            )
        for value in out:
            self._require_int(desc, value)
            if value <= 0:
                raise InvalidScheduleArgumentError(
                    f"{desc} values must be positive, got {value}"
                )
        return out

    def _annotate_split_results(
        self, handle: ir.Value, schedule_ids: list[str]
    ) -> None:
        split = t.SplitHandleOp(
            [self.script.any_op_type] * len(schedule_ids), handle, **self.script.kw
        )
        for idx, schedule_id in enumerate(schedule_ids):
            self.script.annotate_schedule_id(split.results[idx], schedule_id)

    def _transform_error_notes(self, detail: str = "") -> list[str]:
        notes = []
        detail = detail.strip()
        if detail:
            notes.append(text_tail(detail, 40))
        notes.append("transform script:\n" + text_tail(str(self.script.module), 120))
        return notes

    def _partition_kind_name(self, kind) -> str:
        return getattr(kind, "name", str(kind).split(".")[-1])
