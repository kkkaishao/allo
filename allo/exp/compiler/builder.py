# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

from enum import Enum
from .._C import ir
from .._C.ir import (
    Context,
    ModuleOp,
    Value,
    Type,
    AffineMap,
)
from typing import Callable, Sequence, cast, Literal
from .errors import CompilationError
from ..lang.core import (
    APInt,
    DType,
    AlloValue,
    IndexType,
    index,
    ShapedType,
    bool as AlloBool,
    TensorType,
    BufferType,
    StreamType,
    APFloat,
    ConstexprValue,
    TypeBase,
    AlloSymbolRef,
)
from .._C import arith, tensor, linalg, math, memref, allo
from ..lang.rule import get_type_rules


class CmpPred(Enum):
    EQ = 0
    NE = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5


class AlloOpBuilder(ir.AlloOpBuilder):
    """
    IR construction helper used by the MLIR code generator.

    Public `create_*` methods operate on prepared runtime `AlloValue`s. Operator
    lowering is responsible for constexpr materialization, type promotion, and
    broadcasting before calling into this layer. Builder methods use `assert` for
    internal invariants and reserve `compile_error` for user-visible failures
    such as invalid casts, missing type promotion rules, and unbroadcastable
    shapes.
    """

    src: str
    file_name: str | None
    begin_line: int
    curr_node: ast.AST | None
    module: ModuleOp

    def __init__(
        self, context: Context, *, typing_style: Literal["hls", "cpp"] = "hls"
    ):
        super().__init__(context)
        self.typing_style: Literal["hls", "cpp"] = typing_style
        self.type_rules = get_type_rules(typing_style)
        self.src = ""
        self.file_name = None
        self.begin_line = 1
        self.curr_node = None
        self._global_initializer_counter = 0
        self._global_stream_symbols: set[str] = set()

    def compile_error(self, message: str):
        raise CompilationError(
            self.src,
            message,
            self.curr_node,
            file_name=self.file_name,
            begin_line=self.begin_line,
        )

    #####################
    # Constant Creation
    #####################

    def create_const_float(self, value: float, dtype: DType) -> AlloValue:
        ir_ty = dtype.materialize(self.context)
        return AlloValue(arith.ConstantFloatOp(self, ir_ty, value), dtype)

    def create_const_int(self, value: int, dtype: DType) -> AlloValue:
        assert dtype.is_int_signless()
        ir_ty = dtype.materialize(self.context)
        return AlloValue(arith.ConstantIntOp(self, ir_ty, value), dtype)

    def create_const_index(self, value: int) -> AlloValue:
        return AlloValue(arith.ConstantIndexOp(self, value), index)

    def make_scalar(self, value, dtype: DType) -> AlloValue:
        if dtype.is_float():
            return self.create_const_float(value, dtype)
        if dtype.is_int_signless():
            return self.create_const_int(value, dtype)
        if dtype.is_index():
            return self.create_const_index(value)
        return self.compile_error(f"Unsupported scalar type: {dtype}")

    def _fill_shaped_value(self, scalar: AlloValue, shaped: AlloValue) -> AlloValue:
        assert isinstance(shaped.type, ShapedType)
        fill_op = linalg.FillOp(self, scalar.handle, shaped.handle)
        if isinstance(shaped.type, TensorType):
            return AlloValue(fill_op.get_result_at(0), shaped.type)
        assert isinstance(shaped.type, BufferType)
        return shaped

    def _splat_to_shaped(self, scalar: AlloValue, dst_type: ShapedType) -> AlloValue:
        return self._fill_shaped_value(scalar, self.make_buffer(dst_type))

    def materialize_literal_like(self, value, proxy: AlloValue) -> AlloValue:
        if isinstance(proxy.type, DType):
            return self.make_scalar(value, proxy.type)
        assert isinstance(proxy.type, ShapedType)
        return self._fill_shaped_value(self.make_scalar(value, proxy.dtype), proxy)

    def _dense_element_attr(self, value, dtype: DType):
        assert type(value) in (int, float)
        ir_ty = dtype.materialize(self.context)
        if dtype.is_float():
            return ir.FloatAttr.get(ir_ty, float(value))
        if dtype.is_int_signless() or dtype.is_index():
            return ir.IntegerAttr.get(ir_ty, int(value))
        assert False, f"Unsupported dense element type: {dtype}"

    def _next_initializer_name(self, name: str) -> str:
        suffix = self._global_initializer_counter
        self._global_initializer_counter += 1
        return f"{name}_initializer_{suffix}"

    def make_shaped_constant(
        self, values: Sequence[int | float], dst_type: ShapedType, name: str
    ) -> AlloValue:
        num_elements = 1
        for dim in dst_type.shape:
            num_elements *= dim
        assert len(values) == num_elements

        attr_type = TensorType(dst_type.shape, dst_type.dtype).materialize(self.context)
        elements = [self._dense_element_attr(value, dst_type.dtype) for value in values]
        dense_attr = ir.DenseElementsAttr.get(attr_type, elements)
        if isinstance(dst_type, TensorType):
            return AlloValue(arith.ConstantOp(self, dense_attr), dst_type)

        assert isinstance(dst_type, BufferType)
        assert self.module is not None
        memref_type = dst_type.materialize(self.context)
        global_name = self._next_initializer_name(name)
        ip, loc = self.get_insertion_point_and_loc()
        self.set_insertion_point_to_end(self.module.get_body())
        memref.GlobalOp(self, global_name, "private", memref_type, dense_attr, False)
        self.set_insertion_point_and_loc(ip, loc)
        return AlloValue(memref.GetGlobalOp(self, memref_type, global_name), dst_type)

    #####################
    # Stream Creation
    #####################

    def create_global_stream(self, name: str, stream_type: StreamType) -> AlloSymbolRef:
        assert isinstance(stream_type, StreamType) and stream_type.is_global
        if name in self._global_stream_symbols:
            return self.compile_error(f"Global stream '{name}' is already defined")
        assert self.module is not None
        ip, loc = self.get_insertion_point_and_loc()
        self.set_insertion_point_to_end(self.module.get_body())
        allo.GlobalStreamCreateOp(self, name, stream_type.materialize(self.context))
        self.set_insertion_point_and_loc(ip, loc)
        self._global_stream_symbols.add(name)
        return AlloSymbolRef(name, stream_type)

    def create_stream(self, stream_type: StreamType) -> AlloValue:
        assert isinstance(stream_type, StreamType) and not stream_type.is_global
        return AlloValue(
            allo.StreamCreateOp(self, stream_type.materialize(self.context)),
            stream_type,
        )

    def get_global_stream_handle(self, symbol: AlloSymbolRef) -> Value:
        assert isinstance(symbol.type, StreamType)
        return allo.GlobalStreamGetOp(
            self, symbol.type.materialize(self.context), symbol.name
        ).get_result_at(0)

    #####################
    # Type Casting
    #####################

    def create_index_cast(self, value: Value, dst_type: Type) -> Value:
        return arith.IndexCastOp(self, dst_type, value).get_result_at(0)

    def create_ext(
        self,
        value: Value,
        dst_type: Type,
        *,
        signed: bool = True,
        floating: bool = False,
    ) -> Value:
        assert not (signed and floating), "Cannot be both signed and floating"
        if floating:
            return arith.ExtFOp(self, value, dst_type).get_result_at(0)
        else:
            if signed:
                return arith.ExtSIOp(self, value, dst_type).get_result_at(0)
            else:
                return arith.ExtUIOp(self, value, dst_type).get_result_at(0)

    def create_trunc(
        self,
        value: Value,
        dst_type: Type,
        *,
        floating: bool = False,
    ) -> Value:
        if floating:
            return arith.TruncFOp(self, value, dst_type).get_result_at(0)
        else:
            return arith.TruncIOp(self, value, dst_type).get_result_at(0)

    def create_itofp(self, value: Value, dst_type: Type, signed: bool = True) -> Value:
        if signed:
            return arith.SIToFPOp(self, value, dst_type).get_result_at(0)
        else:
            return arith.UIToFPOp(self, value, dst_type).get_result_at(0)

    def create_fptoi(self, value: Value, dst_type: Type, signed: bool = True) -> Value:
        if signed:
            return arith.FPToSIOp(self, value, dst_type).get_result_at(0)
        else:
            return arith.FPToUIOp(self, value, dst_type).get_result_at(0)

    def scalar_cast(self, src: AlloValue, dst_type: DType) -> AlloValue:
        """
        Cast a scalar runtime value to a scalar dtype.
        """
        assert isinstance(src.type, DType)
        src_type = src.type
        value = src.handle
        if src_type == dst_type:
            return src

        dst_ir_type = dst_type.materialize(self.context)

        if src_type.is_int_signless() and dst_type.is_int_signless():
            if src_type.primitive_width < dst_type.primitive_width:
                return AlloValue(
                    self.create_ext(value, dst_ir_type, signed=dst_type.is_int()),
                    dst_type,
                )
            if src_type.primitive_width > dst_type.primitive_width:
                return AlloValue(self.create_trunc(value, dst_ir_type), dst_type)
            return AlloValue(value, dst_type)

        if src_type.is_int_signless() and dst_type.is_index():
            return AlloValue(self.create_index_cast(value, dst_ir_type), dst_type)

        if src_type.is_index() and dst_type.is_int_signless():
            return AlloValue(self.create_index_cast(value, dst_ir_type), dst_type)

        if src_type.is_int_signless() and dst_type.is_float():
            return AlloValue(
                self.create_itofp(value, dst_ir_type, signed=src_type.is_int()),
                dst_type,
            )

        if src_type.is_float() and dst_type.is_int_signless():
            return AlloValue(
                self.create_fptoi(value, dst_ir_type, signed=dst_type.is_int()),
                dst_type,
            )

        if src_type.is_float() and dst_type.is_float():
            if src_type.primitive_width < dst_type.primitive_width:
                return AlloValue(
                    self.create_ext(value, dst_ir_type, signed=False, floating=True),
                    dst_type,
                )
            return AlloValue(
                self.create_trunc(value, dst_ir_type, floating=True), dst_type
            )

        # index2float/float2index is useless
        return self.compile_error(
            f"Unsupported scalar cast from {src_type} to {dst_type}"
        )

    @staticmethod
    def _shaped_type_with_dtype(src_type: ShapedType, dtype: DType) -> ShapedType:
        if isinstance(src_type, TensorType):
            return TensorType(src_type.shape, dtype)
        assert isinstance(src_type, BufferType)
        return BufferType(src_type.shape, dtype)

    def shaped_cast(self, src: AlloValue, dst_type: DType) -> AlloValue:
        """
        Cast a shaped runtime value to the same shape with a new element dtype.
        """
        assert isinstance(src.type, ShapedType) and isinstance(src.dtype, DType)
        if src.dtype == dst_type:
            return src

        handle = tensor.CastOp(
            self, src.handle, dst_type.materialize(self.context)  # TODO: check
        ).get_result_at(0)
        return AlloValue(handle, self._shaped_type_with_dtype(src.type, dst_type))

    def cast_to_dtype(self, src: AlloValue, dtype: DType) -> AlloValue:
        if isinstance(src.type, DType):
            return self.scalar_cast(src, dtype)
        if isinstance(src.type, ShapedType):
            return self.shaped_cast(src, dtype)
        assert False, f"Unsupported value type: {src.type}"

    def _broadcast_shaped_to_type(
        self, src: AlloValue, dst_type: TensorType
    ) -> AlloValue | None:
        assert isinstance(src.type, TensorType)
        shape, indices_src, _ = self.infer_broadcast_shape(
            src.type.shape, dst_type.shape
        )
        if tuple(shape) != tuple(dst_type.shape) or not indices_src:
            return None
        init = tensor.EmptyOp(
            self, shape, dst_type.dtype.materialize(self.context)
        ).get_result_at(0)
        broadcast = linalg.BroadcastOp(self, src.handle, init, indices_src)
        return AlloValue(broadcast, dst_type)

    def cast(self, src: AlloValue | ConstexprValue, dst_type: TypeBase) -> AlloValue:
        assert isinstance(dst_type, TypeBase)
        if isinstance(src, ConstexprValue):
            if isinstance(dst_type, DType):
                return self.make_scalar(src.value, dst_type)
            if isinstance(dst_type, ShapedType):
                return self._splat_to_shaped(
                    self.make_scalar(src.value, dst_type.dtype), dst_type
                )
            assert False, f"Unsupported destination type: {dst_type}"

        assert isinstance(src, AlloValue)
        if isinstance(dst_type, StreamType):
            if src.type == dst_type:
                return src
            return self.compile_error(f"Cannot cast from {src.type} to {dst_type}")

        if isinstance(dst_type, DType):
            return self.cast_to_dtype(src, dst_type)

        if isinstance(src.type, DType) and isinstance(dst_type, ShapedType):
            return self._splat_to_shaped(
                self.scalar_cast(src, dst_type.dtype), dst_type
            )

        if isinstance(src.type, TensorType) and isinstance(dst_type, TensorType):
            if tuple(src.type.shape) == tuple(dst_type.shape):
                return self.shaped_cast(src, dst_type.dtype)
            if src.dtype == dst_type.dtype:
                broadcast = self._broadcast_shaped_to_type(src, dst_type)
                if broadcast is not None:
                    return broadcast

        if isinstance(src.type, BufferType) and isinstance(dst_type, BufferType):
            if src.type == dst_type:
                return src

        return self.compile_error(
            f"Cannot cast from {src.type} to {dst_type}, unsupported type "
            "combination or value is not broadcastable"
        )

    def normalize_indices(
        self,
        indices: Sequence[AlloValue | ConstexprValue | tuple],
        *,
        expected_len: int | None = None,
        context: str | None = None,
    ) -> list[AlloValue]:
        out = []
        for val in indices:
            if isinstance(val, tuple):
                return self.compile_error("Nested tuples are not supported in indices.")
            out.append(self.cast(val, index))

        if expected_len is not None and len(out) != expected_len:
            prefix = f"{context} " if context else ""
            return self.compile_error(
                f"{prefix}expects {expected_len} indices, got {len(out)}."
            )
        return out

    def get_promoted_dtype_nary(
        self,
        op_name: str,
        dtypes: Sequence[DType],
        term_signs: Sequence[int] | None = None,
    ) -> DType:
        if len(dtypes) == 0:
            return self.compile_error("Type promotion requires at least one operand")

        ret = self.type_rules.promote(op_name, dtypes, term_signs=term_signs)
        if ret is not None:
            return ret

        if len(dtypes) == 1:
            operand_desc = f"operand type {dtypes[0]}"
        else:
            operand_desc = "operand types " + ", ".join(str(dtype) for dtype in dtypes)
        return self.compile_error(
            f"No {self.typing_style} type promotion rule for operator "
            f"'{op_name}' with {operand_desc}"
        )

    def get_promoted_dtype(self, lhs: DType, rhs: DType | None, op_name: str):
        if rhs is None:
            ret = self.get_promoted_dtype_nary(op_name, [lhs])
        else:
            signs = [1, -1] if op_name == "sub" else None
            ret = self.get_promoted_dtype_nary(op_name, [lhs, rhs], term_signs=signs)
        return ret

    def _reduce_balanced(
        self,
        operands: Sequence[AlloValue],
        combine: Callable[[AlloValue, AlloValue], AlloValue],
    ) -> AlloValue:
        if len(operands) == 0:
            return self.compile_error("Reduction requires at least one operand")
        curr = list(operands)
        while len(curr) > 1:
            nxt = []
            i = 0
            while i < len(curr):
                if i + 1 < len(curr):
                    nxt.append(combine(curr[i], curr[i + 1]))
                    i += 2
                else:
                    nxt.append(curr[i])
                    i += 1
            curr = nxt
        return curr[0]

    def reduce_balanced(
        self,
        operands: Sequence[AlloValue],
        combine: Callable[[AlloValue, AlloValue], AlloValue],
    ) -> AlloValue:
        return self._reduce_balanced(operands, combine)

    def create_add_nary(
        self, operands: Sequence[AlloValue], *, floating: bool = False
    ) -> AlloValue:
        return self._reduce_balanced(
            operands, lambda lhs, rhs: self.create_add(lhs, rhs, floating=floating)
        )

    def create_sub_nary(
        self,
        operands: Sequence[AlloValue],
        term_signs: Sequence[int],
        *,
        floating: bool = False,
    ) -> AlloValue:
        if len(operands) != len(term_signs):
            return self.compile_error(
                f"Sub reduction expects {len(operands)} signs, got {len(term_signs)}"
            )
        normalized = []
        for operand, sign in zip(operands, term_signs):
            if sign < 0:
                normalized.append(self.create_neg(operand, floating=floating))
            else:
                normalized.append(operand)
        return self.create_add_nary(normalized, floating=floating)

    def create_mul_nary(
        self, operands: Sequence[AlloValue], *, floating: bool = False
    ) -> AlloValue:
        return self._reduce_balanced(
            operands, lambda lhs, rhs: self.create_mul(lhs, rhs, floating=floating)
        )

    ######################
    # Basic arithmetic ops
    ######################
    def _emit_linalg_elementwise_binary(
        self,
        lhs: AlloValue,
        rhs: AlloValue,
        result_type: ShapedType,
        build_fn: Callable,
    ) -> AlloValue:
        assert isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType)
        assert tuple(lhs.type.shape) == tuple(rhs.type.shape)
        res_ir_type = result_type.dtype.materialize(self.context)
        init = tensor.EmptyOp(self, result_type.shape, res_ir_type).get_result_at(0)
        identities = [AffineMap.get_identity(result_type.rank, self.context)] * 3
        iterators = [linalg.PAR] * result_type.rank

        op = linalg.GenericOp(
            self, [res_ir_type], [lhs.handle, rhs.handle], [init], identities, iterators
        )
        body = op.add_entry_block()
        lhs_arg = AlloValue(body.get_arg_at(0), lhs.dtype)
        rhs_arg = AlloValue(body.get_arg_at(1), rhs.dtype)
        ip = self.save_insertion_point()

        try:
            self.set_insertion_point_to_end(body)
            res = build_fn(lhs_arg, rhs_arg)
            linalg.YieldOp(self, [res])
        finally:
            self.restore_insertion_point(ip)

        return AlloValue(op.get_result_at(0), result_type)

    def _emit_linalg_elementwise_unary(
        self,
        operand: AlloValue,
        result_type: ShapedType,
        build_fn: Callable,
    ) -> AlloValue:
        assert isinstance(operand.type, ShapedType)
        res_ir_type = result_type.dtype.materialize(self.context)
        init = tensor.EmptyOp(self, result_type.shape, res_ir_type).get_result_at(0)
        identities = [AffineMap.get_identity(result_type.rank, self.context)] * 2
        iterators = [linalg.PAR] * result_type.rank

        op = linalg.GenericOp(
            self, [res_ir_type], [operand.handle], [init], identities, iterators
        )
        body = op.add_entry_block()
        region_arg = AlloValue(body.get_arg_at(0), operand.dtype)
        ip = self.save_insertion_point()

        try:
            self.set_insertion_point_to_end(body)
            res = build_fn(region_arg)
            linalg.YieldOp(self, [res])
        finally:
            self.restore_insertion_point(ip)

        return AlloValue(op.get_result_at(0), result_type)

    def _emit_elementwise_binary(
        self,
        lhs: AlloValue,
        rhs: AlloValue,
        result_dtype: DType,
        build_fn: Callable,
    ) -> AlloValue:
        lhs_is_shaped = isinstance(lhs.type, ShapedType)
        rhs_is_shaped = isinstance(rhs.type, ShapedType)
        assert lhs_is_shaped == rhs_is_shaped
        if lhs_is_shaped:
            result_type = self._shaped_type_with_dtype(
                cast(ShapedType, lhs.type), result_dtype
            )
            return self._emit_linalg_elementwise_binary(lhs, rhs, result_type, build_fn)
        return AlloValue(build_fn(lhs, rhs), result_dtype)

    def _emit_elementwise_unary(
        self,
        operand: AlloValue,
        result_dtype: DType,
        build_fn: Callable,
    ) -> AlloValue:
        if isinstance(operand.type, ShapedType):
            result_type = self._shaped_type_with_dtype(operand.type, result_dtype)
            return self._emit_linalg_elementwise_unary(operand, result_type, build_fn)
        return AlloValue(build_fn(operand), result_dtype)

    def _emit_binary_op(
        self, lhs: AlloValue, rhs: AlloValue, result_dtype: DType, op_cls
    ) -> AlloValue:
        return self._emit_elementwise_binary(
            lhs,
            rhs,
            result_dtype,
            lambda lhs, rhs: op_cls(self, lhs.handle, rhs.handle).get_result_at(0),
        )

    def create_add(
        self, lhs: AlloValue, rhs: AlloValue, *, floating: bool = False
    ) -> AlloValue:
        op_cls = arith.AddFOp if floating else arith.AddIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_sub(
        self, lhs: AlloValue, rhs: AlloValue, *, floating: bool = False
    ) -> AlloValue:
        op_cls = arith.SubFOp if floating else arith.SubIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_mul(
        self, lhs: AlloValue, rhs: AlloValue, *, floating: bool = False
    ) -> AlloValue:
        op_cls = arith.MulFOp if floating else arith.MulIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_div(
        self,
        lhs: AlloValue,
        rhs: AlloValue,
        *,
        signed: bool = True,
        floating: bool = False,
    ) -> AlloValue:
        assert not (signed and floating)
        if floating:
            op_cls = arith.DivFOp
        elif signed:
            op_cls = arith.DivSIOp
        else:
            op_cls = arith.DivUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_floordiv(
        self,
        lhs: AlloValue,
        rhs: AlloValue,
        *,
        signed: bool = True,
        floating: bool = False,
    ) -> AlloValue:
        assert not (signed and floating)

        def build_fn(lhs, rhs):
            if floating:
                divf = arith.DivFOp(self, lhs.handle, rhs.handle).get_result_at(0)
                return math.FloorOp(self, divf).get_result_at(0)
            elif signed:
                return arith.FloorDivSIOp(self, lhs.handle, rhs.handle).get_result_at(0)
            else:
                # for unsigned, floordiv is the same as div
                return arith.DivUIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        return self._emit_elementwise_binary(lhs, rhs, lhs.dtype, build_fn)

    def create_mod(
        self,
        lhs: AlloValue,
        rhs: AlloValue,
        *,
        signed: bool = True,
        floating: bool = False,
    ) -> AlloValue:
        assert not (signed and floating)
        if floating:
            op_cls = arith.RemFOp
        elif signed:
            op_cls = arith.RemSIOp
        else:
            op_cls = arith.RemUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_pow(
        self,
        base: AlloValue,
        exp: AlloValue,
        *,
        base_floating: bool = False,
        exp_floating: bool = False,
    ) -> AlloValue:
        assert not (
            not base_floating and exp_floating
        ), "If rhs is floating, base must be floating too"

        def build_fn(lhs, rhs):
            if base_floating and exp_floating:
                return math.PowFOp(self, lhs.handle, rhs.handle).get_result_at(0)
            elif base_floating:
                return math.FPowIOp(self, lhs.handle, rhs.handle).get_result_at(0)
            else:
                return math.IPowIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        return self._emit_elementwise_binary(base, exp, base.dtype, build_fn)

    def create_lshift(self, lhs: AlloValue, rhs: AlloValue) -> AlloValue:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.ShLIOp)

    def create_rshift(
        self, lhs: AlloValue, rhs: AlloValue, signed: bool = True
    ) -> AlloValue:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        op_cls = arith.ShRSIOp if signed else arith.ShRUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_neg(self, operand: AlloValue, *, floating: bool = False) -> AlloValue:
        def build_fn(value):
            if floating:
                return arith.NegFOp(self, value.handle).get_result_at(0)
            else:
                # for integer, neg is the same as 0 - operand
                zero = self.make_scalar(0, value.dtype).handle
                return arith.SubIOp(self, zero, value.handle).get_result_at(0)

        return self._emit_elementwise_unary(operand, operand.dtype, build_fn)

    def create_invert(self, operand: AlloValue) -> AlloValue:
        assert isinstance(operand.dtype, APInt)

        def build_fn(value):
            # for integer, invert is the same as -1 xor operand
            ones = 2**value.dtype.primitive_width - 1
            ones_val = self.make_scalar(ones, value.dtype).handle
            return arith.XOrIOp(self, ones_val, value.handle).get_result_at(0)

        return self._emit_elementwise_unary(operand, operand.dtype, build_fn)

    #########################
    # Comparison ops
    #########################

    def create_cmpi(
        self, lhs: AlloValue, rhs: AlloValue, predicate: CmpPred, *, signed=False
    ) -> AlloValue:
        assert isinstance(lhs.dtype, (IndexType, APInt)) and isinstance(
            rhs.dtype, (IndexType, APInt)
        )
        pred_val = predicate.value
        if not signed and predicate in {CmpPred.LT, CmpPred.LE, CmpPred.GT, CmpPred.GE}:
            pred_val += 4  # add offset to get unsigned predicate value

        def build_fn(lhs, rhs):
            return arith.CmpIOp(self, pred_val, lhs.handle, rhs.handle).get_result_at(0)

        return self._emit_elementwise_binary(lhs, rhs, AlloBool, build_fn)

    _cmpf_pred_map = {
        CmpPred.EQ: 1,
        CmpPred.NE: 6,
        CmpPred.LT: 4,
        CmpPred.LE: 5,
        CmpPred.GT: 2,
        CmpPred.GE: 3,
    }

    def create_cmpf(
        self,
        lhs: AlloValue,
        rhs: AlloValue,
        predicate: CmpPred,
        *,
        ordered: bool = False,
    ) -> AlloValue:
        assert isinstance(lhs.dtype, APFloat) and isinstance(rhs.dtype, APFloat)
        pred_val = self._cmpf_pred_map[predicate]
        if ordered and predicate in {CmpPred.LT, CmpPred.LE, CmpPred.GT, CmpPred.GE}:
            pred_val += 7  # add offset to get ordered predicate value

        def build_fn(lhs, rhs):
            return arith.CmpFOp(self, pred_val, lhs.handle, rhs.handle).get_result_at(0)

        return self._emit_elementwise_binary(lhs, rhs, AlloBool, build_fn)

    def create_max(
        self,
        lhs: AlloValue,
        rhs: AlloValue,
        *,
        signed: bool = True,
        floating: bool = False,
        propagate_nan: bool = True,
    ) -> AlloValue:
        assert not (signed and floating)
        if floating:
            op_cls = arith.MaximumFOp if propagate_nan else arith.MaxNumFOp
        elif signed:
            op_cls = arith.MaxSIOp
        else:
            op_cls = arith.MaxUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    def create_min(
        self,
        lhs: AlloValue,
        rhs: AlloValue,
        *,
        signed: bool = True,
        floating: bool = False,
        propagate_nan: bool = True,
    ) -> AlloValue:
        assert not (signed and floating)
        if floating:
            op_cls = arith.MinimumFOp if propagate_nan else arith.MinNumFOp
        elif signed:
            op_cls = arith.MinSIOp
        else:
            op_cls = arith.MinUIOp
        return self._emit_binary_op(lhs, rhs, lhs.dtype, op_cls)

    ##########################
    # Bitwise logical ops
    ##########################

    def create_bitwise_and(self, lhs: AlloValue, rhs: AlloValue) -> AlloValue:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.AndIOp)

    def create_bitwise_or(self, lhs: AlloValue, rhs: AlloValue) -> AlloValue:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.OrIOp)

    def create_bitwise_xor(self, lhs: AlloValue, rhs: AlloValue) -> AlloValue:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.XOrIOp)

    ###########################
    # Logical ops
    ###########################
    def create_logical_and(self, lhs: AlloValue, rhs: AlloValue) -> AlloValue:
        assert lhs.dtype == AlloBool and rhs.dtype == AlloBool
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.AndIOp)

    def create_logical_or(self, lhs: AlloValue, rhs: AlloValue) -> AlloValue:
        assert lhs.dtype == AlloBool and rhs.dtype == AlloBool
        return self._emit_binary_op(lhs, rhs, lhs.dtype, arith.OrIOp)

    def create_logical_not(self, operand: AlloValue) -> AlloValue:
        assert operand.dtype == AlloBool

        def build_fn(value):
            # logical not can be implemented as 1 xor operand
            ones = self.make_scalar(1, value.dtype).handle
            return arith.XOrIOp(self, ones, value.handle).get_result_at(0)

        return self._emit_elementwise_unary(operand, operand.dtype, build_fn)

    ###########################
    # Broadcasting
    ###########################
    @staticmethod
    def infer_broadcast_shape(shape1: Sequence[int], shape2: Sequence[int]):
        """
        infer the broadcasted shape of two tensors, returning

        1. the broadcasted shape,
        2. the indices of dimensions in shape1 that are broadcasted
        3. the indices of dimensions in shape2 that are broadcasted

        return empty lists if the two shapes are not broadcastable

        using numpy broadcasting rules: https://numpy.org/doc/stable/user/basics.broadcasting.html
        """
        res_shape = []
        a_indices = []
        b_indices = []

        len_a = len(shape1)
        len_b = len(shape2)
        max_rank = max(len_a, len_b)
        for i in range(1, max_rank + 1):
            idx = max_rank - i
            # get dimensions from the end, if one shape is shorter, treat missing dimensions as 1
            dim_a = shape1[-i] if i <= len_a else 1
            dim_b = shape2[-i] if i <= len_b else 1
            if dim_a == dim_b:
                res_shape.append(dim_a)
            elif dim_a == 1:
                res_shape.append(dim_b)
                if dim_b > 1:
                    a_indices.append(idx)
            elif dim_b == 1:
                res_shape.append(dim_a)
                if dim_a > 1:
                    b_indices.append(idx)
            else:
                return [], [], []

        res_shape.reverse()
        a_indices.reverse()
        b_indices.reverse()
        return res_shape, a_indices, b_indices

    def broadcast_pair(
        self, lhs: AlloValue, rhs: AlloValue
    ) -> tuple[AlloValue, AlloValue]:
        assert (
            lhs.dtype == rhs.dtype
        ), "Broadcasting requires operands to have the same dtype"
        lhs_is_shaped = isinstance(lhs.type, ShapedType)
        rhs_is_shaped = isinstance(rhs.type, ShapedType)

        if lhs_is_shaped and rhs_is_shaped:
            lhs_shape = cast(ShapedType, lhs.type).shape
            rhs_shape = cast(ShapedType, rhs.type).shape
            shape, indices_lhs, indices_rhs = self.infer_broadcast_shape(
                lhs_shape, rhs_shape
            )
            if not shape:
                return self.compile_error(
                    f"Shapes {lhs_shape} and {rhs_shape} are not broadcastable"
                )
            init = tensor.EmptyOp(
                self, shape, lhs.dtype.materialize(self.context)
            ).get_result_at(0)
            if not indices_lhs and not indices_rhs:
                return lhs, rhs
            if indices_lhs:
                lhs_handle = linalg.BroadcastOp(
                    self, lhs.handle, init, indices_lhs
                ).get_result_at(0)
                if isinstance(lhs.type, TensorType):
                    lhs_type = TensorType(shape, lhs.dtype)
                else:
                    lhs_type = BufferType(shape, lhs.dtype)
                return AlloValue(lhs_handle, lhs_type), rhs
            else:
                rhs_handle = linalg.BroadcastOp(
                    self, rhs.handle, init, indices_rhs
                ).get_result_at(0)
                if isinstance(rhs.type, TensorType):
                    rhs_type = TensorType(shape, rhs.dtype)
                else:
                    rhs_type = BufferType(shape, rhs.dtype)
                return lhs, AlloValue(rhs_handle, rhs_type)

        if not lhs_is_shaped and not rhs_is_shaped:
            # both are scalars, no need to broadcast
            return lhs, rhs

        if isinstance(lhs.type, BufferType) or isinstance(rhs.type, BufferType):
            return self.compile_error("Scalars cannot broadcast to buffer types")
        # one is scalar, it can be broadcasted to any shape
        if lhs_is_shaped and not rhs_is_shaped:
            rhs_handle = tensor.SplatOp(
                self, rhs.handle, cast(ShapedType, lhs.type).shape
            )
            return lhs, AlloValue(rhs_handle, lhs.type)
        else:
            lhs_handle = tensor.SplatOp(
                self, lhs.handle, cast(ShapedType, rhs.type).shape
            )
            return AlloValue(lhs_handle, rhs.type), rhs

    ###########################
    # Memory operations
    ###########################

    def make_buffer(self, type: ShapedType) -> AlloValue:
        if isinstance(type, BufferType):
            alloc_op = memref.AllocOp(self, type.materialize(self.context))
            return AlloValue(alloc_op, type)
        if isinstance(type, TensorType):
            empty_op = tensor.EmptyOp(
                self, type.shape, type.dtype.materialize(self.context)
            )
            return AlloValue(empty_op, type)
        assert False, f"Unsupported shaped type: {type}"

    def fill_buffer(self, buffer: AlloValue, value: AlloValue) -> AlloValue | None:
        assert isinstance(buffer.type, ShapedType)
        if isinstance(buffer.type, TensorType):
            return self._fill_shaped_value(value, buffer)
        assert isinstance(buffer.type, BufferType)
        self._fill_shaped_value(value, buffer)
        return None

    def create_load(self, lhs: AlloValue, indices: Sequence[AlloValue]) -> AlloValue:
        assert isinstance(lhs.type, ShapedType)
        if isinstance(lhs.type, BufferType):
            index_values = [idx.handle for idx in indices]
            load_op = memref.LoadOp(self, lhs.handle, index_values)
            return AlloValue(load_op, lhs.dtype)
        elif isinstance(lhs.type, TensorType):
            index_values = [idx.handle for idx in indices]
            load_op = tensor.ExtractOp(self, lhs.handle, index_values)
            return AlloValue(load_op, lhs.dtype)
        assert False, f"Unsupported shaped type: {lhs.type}"

    def create_store(
        self, value: AlloValue, buffer: AlloValue, indices: Sequence[AlloValue]
    ) -> AlloValue | None:
        assert isinstance(buffer.type, ShapedType)
        if isinstance(buffer.type, BufferType):
            index_values = [idx.handle for idx in indices]
            memref.StoreOp(self, value.handle, buffer.handle, index_values)
            return None
        elif isinstance(buffer.type, TensorType):
            index_values = [idx.handle for idx in indices]
            insert_op = tensor.InsertOp(self, value.handle, buffer.handle, index_values)
            return AlloValue(insert_op, buffer.type)
        assert False, f"Unsupported shaped type: {buffer.type}"

    def _stream_handle_and_indices(
        self, stream: AlloSymbolRef | AlloValue
    ) -> tuple[Value, StreamType, tuple[AlloValue, ...]]:
        assert isinstance(stream, (AlloSymbolRef, AlloValue))
        assert isinstance(stream.type, StreamType)
        assert stream.indices is not None
        if isinstance(stream, AlloSymbolRef):
            return self.get_global_stream_handle(stream), stream.type, stream.indices
        assert not stream.type.is_global
        return stream.handle, stream.type, stream.indices

    def create_stream_get(self, stream: AlloSymbolRef | AlloValue) -> AlloValue:
        handle, stream_type, indices = self._stream_handle_and_indices(stream)
        index_values = [idx.handle for idx in indices]
        get_op = allo.StreamGetOp(self, handle, index_values)
        return AlloValue(get_op, stream_type.base_type)

    def create_stream_put(
        self, stream: AlloSymbolRef | AlloValue, value: AlloValue | ConstexprValue
    ) -> None:
        handle, stream_type, indices = self._stream_handle_and_indices(stream)
        value = self.cast(value, stream_type.base_type)
        index_values = [idx.handle for idx in indices]
        allo.StreamPutOp(self, handle, index_values, value.handle)
        return None
