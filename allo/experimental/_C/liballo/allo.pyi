from enum import Enum
from typing import Sequence, Optional
from . import ir

class PartitionKind(Enum):
    Complete = 0
    Cyclic = 1
    Block = 2

class PartitionAttr(ir.Attribute):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def get(
        context: ir.Context,
        dims: Sequence[int],
        kinds: Sequence[int],
        factors: Sequence[int],
    ) -> PartitionAttr: ...

class CallOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        kernel: KernelOp,
        args: Sequence[ir.Value],
    ) -> CallOp: ...

class KernelOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    def add_entry_block(self) -> ir.Block: ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        name: str,
        type: ir.FunctionType,
        arg_attrs: Sequence[ir.DictionaryAttr] = [],
        virtual_mapping: Sequence[int] | None = None,
    ) -> KernelOp: ...
    def get_arg_at(self, idx: int) -> ir.BlockArgument: ...
    @property
    def func_name(self) -> str: ...
    @property
    def func_type(self) -> ir.FunctionType: ...
    @property
    def num_args(self) -> int: ...
    @property
    def virtual_mapping(self) -> list[int] | None: ...
    @virtual_mapping.setter
    def virtual_mapping(self, map: Sequence[int]) -> None: ...
    def set_arg_attr(self, arg_no: int, name: str, attr: ir.Attribute) -> None: ...
    def set_type(self, type: ir.Type) -> None: ...

class ReturnOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, operands: Sequence[ir.Value]) -> ReturnOp: ...

class ChanGetOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        elt_ty: ir.Type,
        chan: str,
        indices: Sequence[ir.Value],
        blocking: bool = False,
    ) -> ir.Value: ...

class ChanPutOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        chan: str,
        indices: Sequence[ir.Value],
        value: ir.Value,
        blocking: bool = False,
    ) -> ChanPutOp: ...

class ChannelType(ir.Type):
    def __int__(*args, **kwargs): ...
    @staticmethod
    def get(
        context: ir.Context, data_type: ir.Type, capacity: int = 2
    ) -> ChannelType: ...

class ChanAcquireOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        chan: str,
        indices: Sequence[ir.Value],
        tys: ir.Type,
        size: int = 1,
    ) -> ChanAcquireOp: ...

class ChanReleaseOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        chan: str,
        indices: Sequence[ir.Value],
        buffers: Sequence[ir.Value],
    ) -> ChanReleaseOp: ...

class ChanCreateOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        name: str,
        chan_ty: ChannelType,
        shape: Sequence[int] = [],
    ) -> ChanCreateOp: ...

class RaiseToAffineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> RaiseToAffineOp: ...

class OutlineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, target: ir.Value, func_name: str
    ) -> OutlineOp: ...

class TagPipelineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, target: ir.Value, ii: int
    ) -> TagPipelineOp: ...

class TagUnrollOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, target: ir.Value, factor: int
    ) -> TagUnrollOp: ...

class LoopSplitOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, loop: ir.Value, factor: int
    ) -> LoopSplitOp: ...

class LoopTileOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, loops: ir.Value, factors: Sequence[int]
    ) -> LoopTileOp: ...

class LoopFlattenOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, loops: ir.Value) -> LoopFlattenOp: ...

class LoopReorderOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, loops: ir.Value, permutation: Sequence[int]
    ) -> LoopReorderOp: ...

class ComputeAtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, producer: ir.Value, consumer_loop: ir.Value
    ) -> ComputeAtOp: ...

class ReuseAtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, target: ir.Value, axis: ir.Value
    ) -> ReuseAtOp: ...

class RenameOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, target: ir.Value, new_name: str
    ) -> RenameOp: ...

class GetProgramIdOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, dim: int) -> ir.Value: ...

class GetNumProgramsOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, dim: int) -> ir.Value: ...

class BitExtractOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, src: ir.Value, lo: ir.Value, width: int
    ) -> ir.Value: ...

class BitInsertOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        dst: ir.Value,
        src: ir.Value,
        lo: ir.Value,
        width: int,
    ) -> ir.Value: ...

class BitConcatOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, srcs: Sequence[ir.Value]) -> ir.Value: ...

class MatchValueOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, target: ir.Value, name: str, idx: int
    ) -> ir.Value: ...

class PartitionOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, target: ir.Value, partition: PartitionAttr
    ) -> PartitionOp: ...

class ApplyVirtualMapOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        mapping: Sequence[int] = [],
        enable_sccp: bool = False,
    ) -> ApplyVirtualMapOp: ...

class ChainOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder, first: ir.Value, second: ir.Value
    ) -> ChainOp: ...

class BundleOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, inputs: ir.Value) -> BundleOp: ...
