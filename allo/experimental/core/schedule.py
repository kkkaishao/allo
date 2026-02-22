# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .._C.liballo import ir, transform as tran_d, allo as allo_d
from .. import PartitionKind
import inspect as pyinspect
from dataclasses import dataclass
from typing import Dict, List

MatchAttrValue = str | bool | int | float


@dataclass(frozen=True)
class _HandleConsume:
    value: ir.Value


@dataclass(frozen=True)
class _HandleProvide:
    name: str
    value: ir.Value
    select: bool = False


class Schedule:
    sched_mod: ir.ModuleOp
    entry: tran_d.NamedSequenceOp
    entry_block: ir.Block
    root: ir.BlockArgument

    def __init__(self, module: ir.ModuleOp, context: ir.Context):
        self.module = module
        self.context = context
        self.context.load_transform_dialects()
        self.builder = ir.AlloOpBuilder(context)

        # map from handle names to transform handles
        self.sym_table: Dict[str, ir.Value] = {}
        self.active_handle: str | None = None
        self.invalid_handles: set[str] = set()
        self._anon_match_count = 0
        self._init_transform()

    def _init_transform(self):
        # create a new module for transform operations
        self.builder.set_insertion_point_to_end(self.module.body)
        sched_mod = ir.ModuleOp.create(self.builder)
        sched_mod.set_attr(
            "transform.with_named_sequence", ir.UnitAttr.get(self.context)
        )
        self.sched_mod = sched_mod

        # create the entry point for the transform script
        self.builder.set_insertion_point_to_start(sched_mod.body)
        op_ty = tran_d.OperationType.get(self.context, "builtin.module")
        seq = tran_d.NamedSequenceOp.create(self.builder, "__transform_main", op_ty, [])
        self.entry = seq
        self.entry_block = seq.entry_block
        self.root = seq.get_arg_at(0)
        self._register_handle_value("module", self.root, select=True)

        # create yield
        self.builder.set_insertion_point_to_end(self.entry_block)
        tran_d.YieldOp.create(self.builder, [])

        # default insertion point to the start of the sequence
        self.builder.set_insertion_point_to_start(self.entry_block)

    def _next_anonymous_match_name(self):
        name = f"match_{self._anon_match_count}"
        self._anon_match_count += 1
        return name

    def _normalize_op_names(self, op_name: List[str] | str | None):
        if op_name is None:
            return []
        if isinstance(op_name, str):
            return [op_name]
        return op_name

    def _normalize_handle_names(
        self, handles: List[str] | str | None, action: str
    ) -> List[str]:
        if handles is None:
            if self.active_handle is None:
                raise ValueError(
                    f"No target for {action}. Please provide a handle name or perform a match/select first."
                )
            return [self.active_handle]
        if isinstance(handles, str):
            return [handles]
        if len(handles) == 0:
            raise ValueError(f"Empty handle list for {action}.")
        return handles

    def update_loc(
        self, filename: str | None = None, line: int | None = None, col: int = 1
    ):
        """
        Update builder location for subsequent transform op creation.
        If `filename` and `line` are not provided, reset to unknown location.
        """
        if filename is None or line is None:
            self.builder.set_unknown_loc()
            return self
        self.builder.loc = ir.Location(filename, line, col, self.context)
        return self

    def _refresh_builder_loc_from_callsite(self):
        """
        Set builder location to the Python callsite where a schedule API is invoked.
        """
        frame = pyinspect.currentframe()
        try:
            if frame is not None:
                frame = frame.f_back
            internal_file = __file__
            while frame is not None and frame.f_code.co_filename == internal_file:
                frame = frame.f_back
            if frame is None:
                self.builder.set_unknown_loc()
                return
            self.update_loc(frame.f_code.co_filename, frame.f_lineno, 1)
        finally:
            del frame

    def _to_match_attr(self, value: MatchAttrValue):
        if isinstance(value, str):
            return self.builder.get_str_attr(value)
        if isinstance(value, bool):
            return ir.BoolAttr.get(value, self.context)
        if isinstance(value, int):
            i64 = ir.IntegerType.get(64, self.context)
            return ir.IntegerAttr.get(i64, value)
        if isinstance(value, float):
            f32 = ir.F32Type.get(self.context)
            return ir.FloatAttr.get(f32, value)
        raise TypeError(f"Unsupported match attribute type: {type(value)}")

    def _resolve_match_attrs(
        self,
        sym_name: str | None,
        attrs: Dict[str, MatchAttrValue] | None,
    ):
        all_attrs = {}
        if attrs is not None:
            for key, value in attrs.items():
                all_attrs[key] = self._to_match_attr(value)
        if sym_name is not None:
            all_attrs["sym_name"] = self.builder.get_str_attr(sym_name)
        if len(all_attrs) == 0:
            return None
        return self.builder.get_dict_attr(all_attrs)

    def _resolve_target(
        self,
        target: str | None,
        action: str,
        allow_auto_sym_match: bool = False,
    ):
        if target is None:
            if self.active_handle is None:
                raise ValueError(
                    f"No target for {action}. Please provide a handle name or perform a match/select first."
                )
            target = self.active_handle

        if isinstance(target, str):
            if target in self.sym_table:
                if target in self.invalid_handles:
                    raise ValueError(
                        f"Handle '{target}' is invalidated by a previous consuming transform. "
                        "Please rematch from a valid handle (typically 'module')."
                    )
                self.active_handle = target
                return self.sym_table[target]
            if allow_auto_sym_match:
                self.match(sym_name=target, as_name=target)
                return self.sym_table[target]
            raise ValueError(
                f"Handle '{target}' not found in the schedule. "
                "Please match/select it explicitly before applying transforms."
            )

        raise TypeError(f"Unsupported target type for {action}: {type(target)}")

    def _resolve_handle_name(self, handle: str | None, action: str):
        if handle is None:
            if self.active_handle is None:
                raise ValueError(
                    f"No target for {action}. Please provide a handle name or perform a match/select first."
                )
            return self.active_handle
        return handle

    def _handle_aliases(self, value: ir.Value):
        return [name for name, handle in self.sym_table.items() if handle == value]

    def _register_handle_value(self, name: str, value: ir.Value, select: bool = True):
        self.sym_table[name] = value
        if name in self.invalid_handles:
            self.invalid_handles.remove(name)
        if select:
            self.active_handle = name

    def _register_handle(self, name: str, target: str | None = None):
        """
        Rename an existing handle to `name` via transform.rename and update
        schedule tracking accordingly.

        Args:
            name: New handle name.
            target: Existing handle name to rename. If None, uses current active handle.
        """
        src_name = self._resolve_handle_name(target, "_register_handle")
        if src_name == "module":
            raise ValueError("Cannot rename reserved handle 'module' via bind().")
        if src_name == name:
            self.select(src_name)
            return self

        value = self._resolve_target(target, "_register_handle")
        self._refresh_builder_loc_from_callsite()
        allo_d.RenameOp.create(self.builder, value, name)
        self._apply_handle_effects(
            consumes=[_HandleConsume(value=value)],
            provides=[_HandleProvide(name=name, value=value, select=True)],
        )
        return self

    def _invalidate(self, handles: List[str] | str | None = None):
        """
        Mark handles invalid so they can no longer be selected/used.

        Args:
            handles: Handle name(s) to invalidate. If None, invalidates the current
                active handle.
        """
        names = self._normalize_handle_names(handles, "_invalidate")
        for name in names:
            if name == "module":
                raise ValueError("Cannot invalidate reserved handle 'module'.")
            if name not in self.sym_table:
                raise ValueError(f"Handle '{name}' not found in the schedule.")
            self.invalid_handles.add(name)

        if (
            self.active_handle is not None
            and self.active_handle in self.invalid_handles
        ):
            fallback = None
            if "module" in self.sym_table and "module" not in self.invalid_handles:
                fallback = "module"
            else:
                for name in self.sym_table.keys():
                    if name not in self.invalid_handles:
                        fallback = name
                        break
            self.active_handle = fallback
        return self

    def _apply_handle_effects(
        self,
        consumes: List[_HandleConsume],
        provides: List[_HandleProvide] | None = None,
        invalidate_unrelated: bool = False,
    ):
        """
        Apply handle lifecycle updates after a transform op.

        Args:
            consumes: Handles consumed by the transform.
            provides: New handles produced by the transform.
            invalidate_unrelated: If True, invalidate every non-module handle that
                is not in consumed aliases (used by conservative consuming ops).
        """
        consumed_aliases = set()
        for consume in consumes:
            consumed_aliases.update(self._handle_aliases(consume.value))

        to_invalidate = {name for name in consumed_aliases if name != "module"}
        if invalidate_unrelated:
            for name in self.sym_table.keys():
                if name != "module" and name not in consumed_aliases:
                    to_invalidate.add(name)

        if len(to_invalidate) > 0:
            self._invalidate(sorted(to_invalidate))

        if provides is None or len(provides) == 0:
            return

        produced_order: List[str] = []
        produced_values: Dict[str, ir.Value] = {}
        selected_name: str | None = None

        for provide in provides:
            name = provide.name
            value = provide.value
            if name not in produced_values:
                produced_order.append(name)
            produced_values[name] = value
            if provide.select:
                selected_name = name

        for name in produced_order:
            self._register_handle_value(name, produced_values[name], select=False)

        if selected_name is None and len(produced_order) > 0:
            # Default policy for multi-result transforms: select the last provided
            # handle as the current active handle.
            selected_name = produced_order[-1]
        if selected_name is not None:
            self._register_handle_value(
                selected_name, produced_values[selected_name], select=True
            )

    def inspect(self):
        """
        Print the current state of the schedule for debugging.
        """
        print("=== Schedule Transform Module ===")
        print(self.sched_mod)
        if len(self.sym_table) > 0:
            print("=== Schedule Handles ===")
            print(", ".join(self.sym_table.keys()))
            print("")
        if len(self.invalid_handles) > 0:
            print("=== Invalidated Handles ===")
            print(", ".join(sorted(self.invalid_handles)))
            print("")
        return self

    def commit(self):
        """
        Verify and apply the schedule to the module.
        """
        if not self.sched_mod.verify():
            raise ValueError("Schedule module verification failed.")
        failed, err_msg = tran_d.apply_transforms(
            self.module.operation, self.entry.operation, self.sched_mod
        )
        if failed:
            print("Schedule application failed with error:\n")
            print(err_msg)
            raise RuntimeError("Schedule application failed.")
        if not self.module.verify():
            print(self.module)
            raise RuntimeError("Module verification failed after applying schedule.")

    def match(
        self,
        sym_name: str | None = None,
        inside: str | None = None,
        op_name: List[str] | str | None = None,
        res_name: str | None = None,
        attrs: Dict[str, MatchAttrValue] | None = None,
        as_name: str | None = None,
    ):
        """
        Match operations in the transform module and bind the resulting handle.

        If `sym_name` is provided, it is translated to an op attr filter `sym_name = "..."`
        If `inside` is provided, only match operations inside that handle.
        If `op_name` is provided, only match operations with those op names.
        If `attrs` is provided, all key-value pairs are added to op attr filters.
        `as_name` controls the handle name stored in the schedule. If omitted, fallback order is:
        `sym_name`, then an auto-generated name.

        Args:
            sym_name: Optional symbol name filter (`sym_name` attribute in payload IR).
            inside: Optional parent handle to scope the match.
            op_name: Optional op name or op-name list filter.
            res_name: Optional expected result op type name for the match handle.
            attrs: Optional extra attribute filters (frontend scalar types only).
            as_name: Handle name to register for this match result.
        """
        if inside is not None:
            target = self._resolve_target(inside, "match")
        else:
            target = self.root

        dict_attr = self._resolve_match_attrs(sym_name, attrs)

        if res_name is not None:
            res_ty = tran_d.OperationType.get(self.context, res_name)
        else:
            res_ty = tran_d.AnyOpType.get(self.context)

        op_names = self._normalize_op_names(op_name)

        self._refresh_builder_loc_from_callsite()
        match = tran_d.MatchOp.create(self.builder, target, res_ty, op_names, dict_attr)

        # update trackers
        handle_name = as_name if as_name is not None else sym_name
        if handle_name is None:
            handle_name = self._next_anonymous_match_name()
        self._register_handle_value(handle_name, match, select=True)
        return self

    def match_ops(
        self,
        op_name: List[str] | str,
        inside: str | None = None,
        res_name: str | None = None,
        attrs: Dict[str, MatchAttrValue] | None = None,
        as_name: str | None = None,
    ):
        """
        Convenience wrapper to match a class of operations by op name and attrs.

        Args:
            op_name: Op name or list of op names to match.
            inside: Optional parent handle to scope the match.
            res_name: Optional expected result op type name.
            attrs: Optional attribute filters.
            as_name: Handle name to register for this match result.
        """
        return self.match(
            sym_name=None,
            inside=inside,
            op_name=op_name,
            res_name=res_name,
            attrs=attrs,
            as_name=as_name,
        )

    def select(self, handle: str):
        """
        Set the current active handle for subsequent transformations.

        Args:
            handle: Existing valid handle name.
        """
        self._resolve_target(handle, "select")
        return self

    def bind(self, name: str, target: str | None = None):
        """
        Rename a handle to `name` and invalidate the old handle name.

        Args:
            name: New handle name written to payload op `sym_name`.
            target: Existing handle name to rename. If None, renames current active handle.
        """
        return self._register_handle(name, target)

    #########################
    # Generic Transformations
    #########################

    def cse(self, sym_name: str | None = None):
        """
        Apply CSE to a handle.
        - `None`: current active handle
        - `str`: handle name; fallback to auto `sym_name` match
        """
        target = self._resolve_target(sym_name, "CSE", allow_auto_sym_match=True)
        self._refresh_builder_loc_from_callsite()
        tran_d.ApplyCSEOp.create(self.builder, target)
        return self

    def dce(self, sym_name: str | None = None):
        """
        Apply DCE to a handle.
        """
        target = self._resolve_target(sym_name, "DCE", allow_auto_sym_match=True)
        self._refresh_builder_loc_from_callsite()
        tran_d.ApplyDCEOp.create(self.builder, target)
        return self

    _pattern_map = {"canonicalize": tran_d.ApplyCanonicalizationOp}

    def apply_patterns(self, patterns: List[str] | str, sym_name: str | None = None):
        """
        Apply a pattern list to a handle.
        """
        target = self._resolve_target(
            sym_name, "applying patterns", allow_auto_sym_match=True
        )

        if isinstance(patterns, str):
            patterns = [patterns]

        # apply patterns:
        # transform.apply_patterns to %target { // entry
        #   // pattern ops
        #   transform.apply_patterns.canonicalization
        # } : !transform.any_op
        self._refresh_builder_loc_from_callsite()
        entry = tran_d.ApplyPatternsOp.create(self.builder, target).body
        ip = self.builder.save_insertion_point()
        self.builder.set_insertion_point_to_start(entry)
        for pattern in patterns:
            op = self._pattern_map.get(pattern, None)
            if op is None:
                raise ValueError(f"Unsupported pattern: {pattern}")
            self._refresh_builder_loc_from_callsite()
            op.create(self.builder)
        self.builder.restore_insertion_point(ip)

        return self

    def canonicalize(self, sym_name: str | None = None):
        """
        Apply canonicalization patterns to a handle.
        """
        return self.apply_patterns("canonicalize", sym_name)

    def licm(self, sym_name: str | None = None):
        """
        Apply LICM to a handle. The handle must refer to loop operations.
        """
        target = self._resolve_target(sym_name, "LICM", allow_auto_sym_match=True)
        self._refresh_builder_loc_from_callsite()
        tran_d.ApplyLICMOp.create(self.builder, target)
        return self

    #########################
    # Loop Transformations
    #########################
    def to_affine(self, sym_name: str | None = None):
        """
        Raise the loop handle to affine loop. The handle must refer to loop operations.
        """
        target = self._resolve_target(
            sym_name, "raising to affine", allow_auto_sym_match=True
        )
        handle_name = self._resolve_handle_name(sym_name, "raising to affine")
        self._refresh_builder_loc_from_callsite()
        raise_op = allo_d.RaiseToAffineOp.create(self.builder, target)
        if raise_op.num_results != 1:
            raise RuntimeError(
                f"Unexpected results from RaiseToAffineOp: {raise_op.num_results}"
            )
        new_target = raise_op.get_result_at(0)
        self._apply_handle_effects(
            consumes=[_HandleConsume(value=target)],
            provides=[_HandleProvide(name=handle_name, value=new_target, select=True)],
        )
        return self

    def pipeline(self, sym_name: str | None = None, ii: int = 1):
        """
        Pipeline the loop handle. The handle must refer to loop operations.
        """
        target = self._resolve_target(sym_name, "pipelining", allow_auto_sym_match=True)
        self._refresh_builder_loc_from_callsite()
        allo_d.TagPipelineOp.create(self.builder, target, ii)
        # does not consume
        return self

    def unroll(
        self, sym_name: str | None = None, factor: int = 1, tag_only: bool = True
    ):
        """
        Unroll the loop handle. The handle must refer to loop operations.
        If `tag_only` is True, only tag the loop with unroll factor without actually unrolling.
        This allows downstream patterns or codegen to decide how to use the unroll factor.
        """
        target = self._resolve_target(sym_name, "unrolling", allow_auto_sym_match=True)
        self._refresh_builder_loc_from_callsite()
        if tag_only:
            allo_d.TagUnrollOp.create(self.builder, target, factor)
            # does not consume
        else:
            # use upstream unroll transform
            tran_d.LoopUnrollOp.create(self.builder, target, factor)
            self._apply_handle_effects(consumes=[_HandleConsume(value=target)])
        return self

    def split(self, sym_name: str | None = None, factor: int = 1):
        """
        Split the loop handle. The handle must refer to loop operations.
        Produces `<base>.outer` and `<base>.inner`; active handle becomes `<base>.inner`.

        Args:
            sym_name: Loop handle name. If None, uses current active handle.
            factor: Positive split factor.
        """
        base_name = self._resolve_handle_name(sym_name, "loop splitting")
        target = self._resolve_target(
            sym_name, "loop splitting", allow_auto_sym_match=True
        )
        self._refresh_builder_loc_from_callsite()
        split_op = allo_d.LoopSplitOp.create(self.builder, target, factor)
        handles = tran_d.SplitHandleOp.create(
            self.builder, split_op.get_result_at(0), 2
        )
        outer = handles.get_result_at(0)
        inner = handles.get_result_at(1)
        self._apply_handle_effects(
            consumes=[_HandleConsume(value=target)],
            provides=[
                _HandleProvide(name=f"{base_name}.outer", value=outer),
                _HandleProvide(name=f"{base_name}.inner", value=inner, select=True),
            ],
        )
        return self

    def outline(self, sym_name: str | None = None, *, func_name: str):
        """
        Outline a single operation or a loop into a new function.
        The handle must refer to loop operations.
        Produces `<func_name>` and `<func_name>.call`; active handle becomes `<func_name>.call`.

        Args:
            sym_name: Handle to outline. If None, uses current active handle.
            func_name: Outlined kernel symbol name (required).
        """
        if len(func_name) == 0:
            raise ValueError("outline requires explicit `func_name`.")
        target = self._resolve_target(sym_name, "outlining", allow_auto_sym_match=True)
        base_name = self._resolve_handle_name(sym_name, "outlining")
        self._refresh_builder_loc_from_callsite()
        out_op = allo_d.OutlineOp.create(self.builder, target, func_name)
        handles = tran_d.SplitHandleOp.create(self.builder, out_op.get_result_at(0), 2)
        func = handles.get_result_at(0)
        call = handles.get_result_at(1)
        self._apply_handle_effects(
            consumes=[_HandleConsume(value=target)],
            provides=[
                _HandleProvide(name=f"{base_name}", value=func),
                _HandleProvide(name=f"{base_name}.call", value=call, select=True),
            ],
        )
        return self

    def tile(
        self, sym_name: List[str] | str | None = None, factors: List[int] | int = 1
    ):
        """
        Tile the loop handle. The handle must refer to loop operations.
        If `factors` is an int, it is applied to the innermost loop. If it is a list, it is applied to loops in depth order.
        Produces `<base>.tile` and `<base>.point` per input handle; no active handle is set by default.

        Args:
            sym_name: One handle or a handle list to tile together.
            factors: Tile factors (single int or per-loop list).
        """
        names = self._normalize_handle_names(sym_name, "tiling")
        pairs: List[tuple[str, ir.Value]] = []
        for name in names:
            target = self._resolve_target(name, "tiling", allow_auto_sym_match=True)
            if all(existing != target for _, existing in pairs):
                pairs.append((name, target))
        base_names = [name for name, _ in pairs]
        targets = [target for _, target in pairs]
        if len(targets) == 0:
            raise ValueError("No valid handle for tiling.")
        if isinstance(factors, int):
            factors = [factors]
        self._refresh_builder_loc_from_callsite()
        merged = tran_d.MergeHandlesOp.create(
            self.builder, targets, deduplicate=True
        ).get_result_at(0)
        tiled = allo_d.LoopTileOp.create(self.builder, merged, factors)
        # tiling consumes the original handle and produces two new groups of handles for the tile loops and the point loops.
        handles = tran_d.SplitHandleOp.create(self.builder, tiled.get_result_at(0), 2)
        tile_loops = handles.get_result_at(0)
        point_loops = handles.get_result_at(1)
        split_tile = tran_d.SplitHandleOp.create(
            self.builder, tile_loops, len(base_names)
        )
        split_point = tran_d.SplitHandleOp.create(
            self.builder, point_loops, len(base_names)
        )
        tiles = [split_tile.get_result_at(i) for i in range(len(base_names))]
        points = [split_point.get_result_at(i) for i in range(len(base_names))]
        provides = [
            _HandleProvide(name=f"{base_names[i]}.tile", value=tiles[i])
            for i in range(len(base_names))
        ]
        provides.extend(
            [
                _HandleProvide(
                    name=f"{base_names[i]}.point",
                    value=points[i],
                )
                for i in range(len(base_names))
            ]
        )
        self._apply_handle_effects(
            consumes=[_HandleConsume(value=targets[i]) for i in range(len(base_names))],
            provides=provides,
        )
        return self

    def flatten(self, sym_name: List[str] | str | None = None):
        """
        Flatten perfectly nested loops in the handle into a single loop. The handle must refer to loop operations.

        Args:
            sym_name: One handle or a handle list for flattening.
        """
        names = self._normalize_handle_names(sym_name, "flattening")
        pairs: List[tuple[str, ir.Value]] = []
        for name in names:
            target = self._resolve_target(name, "flattening", allow_auto_sym_match=True)
            if all(existing != target for _, existing in pairs):
                pairs.append((name, target))
        base_names = [name for name, _ in pairs]
        targets = [target for _, target in pairs]
        if len(targets) == 0:
            raise ValueError("No valid handle for flattening.")
        self._refresh_builder_loc_from_callsite()
        merge = tran_d.MergeHandlesOp.create(
            self.builder, targets, deduplicate=True
        ).get_result_at(0)
        flatten_op = allo_d.LoopFlattenOp.create(self.builder, merge)
        flattened = flatten_op.get_result_at(0)
        self._apply_handle_effects(
            consumes=[_HandleConsume(value=targets[i]) for i in range(len(base_names))],
            provides=[
                _HandleProvide(
                    name=f"{base_names[0]}.flat", value=flattened, select=True
                )
            ],
        )
        return self

    def compute_at(
        self,
        target_sym: str,
        axis_sym: str,
    ):
        """
        Compute the operations in the target handle at the loop level of the axis handle. Both handles must refer to loop operations.

        Args:
            target_sym: Producer/target handle to move (consumed).
            axis_sym: Loop handle where target is attached (read-only).
        """
        target = self._resolve_target(
            target_sym, "compute_at target", allow_auto_sym_match=True
        )
        axis = self._resolve_target(
            axis_sym, "compute_at axis", allow_auto_sym_match=True
        )
        self._refresh_builder_loc_from_callsite()
        allo_d.ComputeAtOp.create(self.builder, target, axis)
        self._apply_handle_effects(consumes=[_HandleConsume(value=target)])
        return self

    def reuse_at(
        self,
        producer: str,
        consumer_loop: str,
    ):
        """
        Reuse the buffer of the operations in the target handle at the loop level of the axis handle. Both handles must refer to loop operations.

        Args:
            producer: Producer handle (consumed).
            consumer_loop: Consumer loop handle (consumed and replaced by result handle).
        """
        consumer_name = self._resolve_handle_name(
            consumer_loop, "reuse_at consumer_loop"
        )
        target = self._resolve_target(
            producer, "reuse_at producer", allow_auto_sym_match=True
        )
        axis = self._resolve_target(
            consumer_loop, "reuse_at consumer_loop", allow_auto_sym_match=True
        )
        self._refresh_builder_loc_from_callsite()
        reuse_op = allo_d.ReuseAtOp.create(self.builder, target, axis)
        consumer = reuse_op.get_result_at(0)
        self._apply_handle_effects(
            consumes=[
                _HandleConsume(value=target),
                _HandleConsume(value=axis),
            ],
            provides=[
                _HandleProvide(name=consumer_name, value=consumer, select=True),
            ],
        )
        return self

    ########################
    # Buffer Transformations
    ########################
    def partition(
        self,
        sym_name: str | None = None,
        indices: List[int] | int = 0,
        dim: int = 0,
        kind: PartitionKind = PartitionKind.Complete,
        factor: int = 0,
    ):
        """
        Partition the buffer with given `dim`, `kind`, and `factor`.

        `sym_name` is the handle to the buffer allocation.
        If `sym_name` points to a kernel, the `indices`-th input buffer of the kernel will be partitioned.
        If `sym_name` points to an allocation with multiple buffers, the `indices`-th buffer will be partitioned.

        :param kind: Partition kind. Can be `Complete`, `Cyclic`, or `Block`.
        :param dim: Dimension to partition.
        :param factor: Partition factor. Must be 0 if `kind` is `Complete`. Must be > 0 if `kind` is `Cyclic` or `Block`.
        """
        target = self._resolve_target(
            sym_name, "buffer partitioning", allow_auto_sym_match=True
        )
        name = self._resolve_handle_name(sym_name, "buffer partitioning")
        if isinstance(indices, int):
            indices = [indices]
        # prepare partition attribute
        if kind == PartitionKind.Complete and factor != 0:
            raise ValueError("Complete partition cannot have non-zero factor.")
        if kind != PartitionKind.Complete and factor <= 0:
            raise ValueError(f"{kind} partition must have positive factor, got {factor}.")
        part = allo_d.PartitionAttr.get(self.context, [dim], [kind.value], [factor])
        self._refresh_builder_loc_from_callsite()
        for idx in indices:
            if idx < 0:
                raise ValueError(f"Partition target indices must be non-negative, got {idx}.")
            value = allo_d.MatchValueOp.create(self.builder, target, name, idx)
            allo_d.PartitionOp.create(self.builder, value, part)
        # does not consume or produce new handles
        return self

