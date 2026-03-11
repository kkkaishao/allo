from enum import Enum
from typing import Sequence

from . import ir

class OperationType(ir.Type):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def get(context: ir.Context, op_name: str) -> OperationType: ...

class ParamType(ir.Type):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def get(context: ir.Context, type: ir.Type) -> ParamType: ...

class AnyOpType(ir.Type):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def get(context: ir.Context) -> AnyOpType: ...

class AnyParamType(ir.Type):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def get(context: ir.Context) -> AnyParamType: ...

class NamedSequenceOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        name: str,
        root_type: ir.Type,
        result_types: Sequence[ir.Type],
    ) -> NamedSequenceOp: ...
    @property
    def entry_block(self) -> ir.Block: ...
    def get_arg_at(self, index: int) -> ir.BlockArgument: ...

class YieldOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, operands: Sequence[ir.Value]) -> YieldOp: ...

class ApplyCSEOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ApplyCSEOp: ...

class ApplyDCEOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ApplyDCEOp: ...

class ApplyCanonicalizationOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder) -> ApplyCanonicalizationOp: ...

class ApplyLICMOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ApplyLICMOp: ...

class ApplyPatternsOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ApplyPatternsOp: ...
    @property
    def body(self) -> ir.Block: ...

class ApplyRegisteredPassOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        pass_name: str,
        pass_options: ir.DictionaryAttr,
        dynamic_args: Sequence[ir.Value],
    ) -> ApplyRegisteredPassOp: ...

class MatchOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        res_type: ir.Type,
        op_names: Sequence[str],
        op_attrs: ir.DictionaryAttr | None = None,
    ) -> ir.Value: ...

class PartitionKind(Enum):
    Complete = ...
    Block = ...
    Cyclic = ...

Complete = PartitionKind.Complete
Block = PartitionKind.Block
Cyclic = PartitionKind.Cyclic

class PartitionAttr(ir.Attribute):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def get(
        context: ir.Context,
        sub_partitions: Sequence[tuple[int, int, int]],
    ) -> PartitionAttr: ...

class LoopUnrollOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> LoopUnrollOp: ...

class MergeHandlesOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        handles: Sequence[ir.Value],
        deduplicate: bool = True,
    ) -> ir.Value: ...

class SplitHandleOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        handle: ir.Value,
        num_splits: int,
    ) -> SplitHandleOp: ...

class RenameOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        name: str,
    ) -> RenameOp: ...

class RaiseToAffineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ir.Value: ...

class OutlineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        kernel_name: str,
    ) -> OutlineOp: ...

class TagPipelineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        ii: int,
    ) -> TagPipelineOp: ...

class TagUnrollOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> TagUnrollOp: ...

class LoopReorderOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        order: Sequence[int],
    ) -> LoopReorderOp: ...

class LoopSplitOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> LoopSplitOp: ...

class LoopTileOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factors: Sequence[int],
    ) -> LoopTileOp: ...

class LoopFlattenOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ir.Value: ...

class ReuseAtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        axis: ir.Value,
    ) -> ReuseAtOp: ...

class ComputeAtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        producer: ir.Value,
        consumer_loop: ir.Value,
    ) -> ComputeAtOp: ...

class BufferAtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        consumer_loop: ir.Value,
    ) -> BufferAtOp: ...

class MatchValueOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        index: int,
        source_kind: int = 0,
    ) -> ir.Value: ...

class PartitionOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        partition: PartitionAttr,
    ) -> PartitionOp: ...

def apply_transforms(
    payload: ir.Operation,
    transform_root: ir.Operation,
    transform_module: ir.ModuleOp,
) -> tuple[bool, str]: ...
