# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

from .._C import ir
from .._C.ir import (
    Context,
    ModuleOp,
    F16Type,
    F32Type,
    F64Type,
    BF16Type,
    IndexType as MLIRIndexType,
    Value,
    Type,
    AffineMap,
)
from typing import Callable, Sequence, cast, Literal
from enum import Enum
from .errors import CompilationError
from ..core.types import (
    APInt,
    DType,
    Proxy,
    IndexType,
    index,
    ShapedType,
    int1,
    TensorType,
    BufferType,
    APFloat,
    Constexpr,
    Stream,
    BaseType,
)
from .._C import arith, tensor, linalg, math, memref, allo
from ..core.rule import TypeTable, CppTypeTable


class CmpPred(Enum):
    EQ = 0
    NE = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5


class AlloOpBuilder(ir.AlloOpBuilder):
    """
    A class for building IR operations in allo compiler, it provides modular methods and utilities for creating IR operations, and also handles error reporting and source code mapping.

    Ideal signatures for the methods in this class should be in terms of frontend proxy values (e.g. `tensor`
    and `constexpr`) and frontend types (e.g. `scalar_type`), and the methods should return frontend proxy values. See `tensor` in `core/types.py` for definitions.

    To raise an error during compilation, use the `compile_error` method. Users don't need to worry about
    error-to-source mapping, the builder will handle that for you. Just make sure to call `compile_error` with a descriptive error message when you encounter an error condition.

    Users don't need to restore insertion point after constructing an IR operation, `CodeGenerator` will handle that for you. Just focus on creating the IR operations with the builder methods, and let `CodeGenerator` manage the control flow and insertion points.

    The builder and `CodeGenerator` together manage the state of the IR generation, and the builder is responsible for providing a clean and user-friendly interface for creating IR operations, while the `CodeGenerator` is responsible for constructing control flow and managing the overall structure of the generated code.

    They manage to separate concerns of IR generation and code structure, making the codebase more modular and easier to maintain.
    """

    src: str
    curr_node: ast.AST | None
    module: ModuleOp

    def __init__(
        self, context: Context, *, typing_style: Literal["hls", "cpp"] = "hls"
    ):
        super().__init__(context)
        self.typing_style: Literal["hls", "cpp"] = typing_style

    def compile_error(self, message: str):
        raise CompilationError(self.curr_node, message, self.src)

    #####################
    # Constant Creation
    #####################

    def create_const_float(self, value: float, dtype: DType) -> Proxy:
        ir_ty = dtype.to_mlir(self.context)
        assert isinstance(ir_ty, (F16Type, F32Type, F64Type, BF16Type))
        return Proxy(arith.ConstantFloatOp(self, ir_ty, value), dtype)

    def create_const_int(self, value: int, dtype: DType) -> Proxy:
        ir_ty = dtype.to_mlir(self.context)
        assert isinstance(ir_ty, ir.IntegerType)
        return Proxy(arith.ConstantIntOp(self, ir_ty, value), dtype)

    def create_const_index(self, value: int) -> Proxy:
        return Proxy(arith.ConstantIndexOp(self, value), IndexType())

    def make_scalar(self, value, dtype: DType) -> Proxy:
        if dtype.is_float():
            return self.create_const_float(value, dtype)
        if dtype.is_int():
            return self.create_const_int(value, dtype)
        if dtype.is_index():
            return self.create_const_index(value)
        self.compile_error(f"Unsupported scalar type: {dtype}")

    def make_scalar_or_shaped(self, value, proxy: Proxy) -> Proxy | None:
        type = proxy.type
        if isinstance(type, DType):
            return self.make_scalar(value, type)
        elif isinstance(type, ShapedType):
            const = self.make_scalar(value, type.dtype)
            fill_op = linalg.FillOp(self, const.handle, proxy.handle)
            if isinstance(proxy.type, TensorType):
                return Proxy(fill_op.get_result_at(0), proxy.type)
            else:
                # memref mode has no return vals
                return None

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

    def scalar_cast(self, src: Proxy, dst_type: DType) -> Proxy:
        """
        Perform scalar type casting, user must ensure that `src` is a scalar and `dst_ty` is a scalar type. The method will return a new proxy value with the same value as `src` but with type `dst_ty`.

        If `src` is already of type `dst_ty`, it will return `src` directly.
        """
        assert isinstance(src.type, DType)
        src_type = src.type
        value = src.handle
        if src_type == dst_type:
            return src

        # int to int
        if src_type.is_int() and dst_type.is_int():
            int_ty = dst_type.to_mlir(self.context)
            if src_type.primitive_width < dst_type.primitive_width:
                return Proxy(self.create_ext(value, int_ty), dst_type)
            else:
                return Proxy(self.create_trunc(value, int_ty), dst_type)

        # int to uint
        if src_type.is_int() and dst_type.is_uint():
            int_ty = dst_type.to_mlir(self.context)
            if src_type.primitive_width < dst_type.primitive_width:
                return Proxy(self.create_ext(value, int_ty, signed=False), dst_type)
            else:
                if src_type.primitive_width == dst_type.primitive_width:
                    # reinterpret cast
                    return Proxy(value, dst_type)
                return Proxy(self.create_trunc(value, int_ty), dst_type)

        # int to index
        if src_type.is_int() and dst_type.is_index():
            index_ty = MLIRIndexType.get(self.context)
            return Proxy(self.create_index_cast(value, index_ty), dst_type)

        # int to float
        if src_type.is_int() and dst_type.is_float():
            fp_ty = dst_type.to_mlir(self.context)
            return Proxy(self.create_itofp(value, fp_ty, signed=True), dst_type)

        # uint to int
        if src_type.is_uint() and dst_type.is_int():
            int_ty = dst_type.to_mlir(self.context)
            if src_type.primitive_width < dst_type.primitive_width:
                return Proxy(self.create_ext(value, int_ty), dst_type)
            else:
                if src_type.primitive_width == dst_type.primitive_width:
                    # reinterpret cast
                    return Proxy(value, dst_type)
                return Proxy(self.create_trunc(value, int_ty), dst_type)

        # uint to uint
        if src_type.is_uint() and dst_type.is_uint():
            int_ty = dst_type.to_mlir(self.context)
            if src_type.primitive_width < dst_type.primitive_width:
                return Proxy(self.create_ext(value, int_ty, signed=False), dst_type)
            else:
                return Proxy(self.create_trunc(value, int_ty), dst_type)

        # uint to index
        if src_type.is_uint() and dst_type.is_index():
            index_ty = MLIRIndexType.get(self.context)
            return Proxy(self.create_index_cast(value, index_ty), dst_type)

        # uint to float
        if src_type.is_uint() and dst_type.is_float():
            fp_ty = dst_type.to_mlir(self.context)
            return Proxy(self.create_itofp(value, fp_ty, signed=False), dst_type)

        # float to int
        if src_type.is_float() and dst_type.is_int():
            int_ty = dst_type.to_mlir(self.context)
            return Proxy(self.create_fptoi(value, int_ty, signed=True), dst_type)

        # float to uint
        if src_type.is_float() and dst_type.is_uint():
            int_ty = dst_type.to_mlir(self.context)
            return Proxy(self.create_fptoi(value, int_ty, signed=False), dst_type)

        # float to float
        if src_type.is_float() and dst_type.is_float():
            fp_ty = dst_type.to_mlir(self.context)
            if src_type.primitive_width < dst_type.primitive_width:
                return Proxy(self.create_ext(value, fp_ty, floating=True), dst_type)
            else:
                return Proxy(self.create_trunc(value, fp_ty, floating=True), dst_type)

        # index to int/uint
        if src_type.is_index() and dst_type.is_int_signless():
            int_ty = dst_type.to_mlir(self.context)
            return Proxy(self.create_index_cast(value, int_ty), dst_type)

        # index2float/float2index is useless
        self.compile_error(f"Unsupported scalar cast from {src_type} to {dst_type}")

    def tensor_cast(self, src: Proxy, dst_type: DType) -> Proxy:
        """
        Perform tensor type casting. It creates a new tensor with the same shape as `src` but with data type `dst_dtype`.
        """
        assert isinstance(src.type, ShapedType) and isinstance(src.dtype, DType)
        src_dtype = src.dtype
        value = src.handle
        if src_dtype == dst_type:
            return src

        handle = tensor.CastOp(
            self, value, dst_type.to_mlir(self.context)
        ).get_result_at(0)
        if isinstance(src.type, TensorType):
            new_type = TensorType(dst_type, src.type.shape)
        elif isinstance(src.type, BufferType):
            new_type = BufferType(dst_type, src.type.shape)
        else:
            new_type = ShapedType(dst_type, src.type.shape)
        return Proxy(handle, new_type)

    def as_condition_scalar(self, cond: Proxy, *, kind: str = "if") -> Proxy:
        if isinstance(cond.type, ShapedType):
            if kind == "ifexp":
                self.compile_error(
                    "Condition of ternary expression cannot be a shaped type."
                )
            self.compile_error("Condition of 'if' statement cannot be a shaped type.")
        return self.scalar_cast(cond, int1)

    def cast(self, src: Proxy | Constexpr, dst_type: BaseType) -> Proxy:
        # Case 1: src is constexpr
        if isinstance(src, Constexpr):
            # Case 1.1: dst_type is DType
            if isinstance(dst_type, DType):
                return self.make_scalar(src.value, dst_type)
            # Case 1.2: dst_type is ShapedType
            if isinstance(dst_type, ShapedType):
                const = self.make_scalar(src.value, dst_type.dtype)
                buffer = self.make_buffer(dst_type)
                fill_op = linalg.FillOp(self, const.handle, buffer.handle)
                if isinstance(dst_type, TensorType):
                    return Proxy(fill_op, dst_type)
                if isinstance(dst_type, BufferType):
                    return buffer
        else:
            assert isinstance(src, Proxy)
            # Case 2: src.type is DType
            if isinstance(src.type, DType):
                # Case 2.1: dst_type is DType
                if isinstance(dst_type, DType):
                    return self.scalar_cast(src, dst_type)
                # Case 2.2: dst_type is ShapedType
                if isinstance(dst_type, ShapedType):
                    const = self.scalar_cast(src, dst_type.dtype)
                    buffer = self.make_buffer(dst_type)
                    fill_op = linalg.FillOp(self, const.handle, buffer.handle)
                    if isinstance(dst_type, TensorType):
                        return Proxy(fill_op, dst_type)
                    if isinstance(dst_type, BufferType):
                        return buffer
            # Case 3: src.type is TensorType
            if isinstance(src.type, TensorType):
                # Case 3.1: dst_type is TensorType
                if isinstance(dst_type, TensorType):
                    # Case 3.1.1: same shape, dtype may differ
                    if src.shape == dst_type.shape:
                        return self.tensor_cast(src, dst_type.dtype)
                    # Case 3.1.2: same dtype, shape may differ but broadcastable
                    if src.dtype == dst_type.dtype:
                        shape, indices_a, _ = self.infer_broadcast_shape(
                            src.type.shape, dst_type.shape
                        )
                        if shape and indices_a:
                            # need broadcasting
                            init = tensor.EmptyOp(
                                self, shape, dst_type.dtype.to_mlir(self.context)
                            ).get_result_at(0)
                            broadcast = linalg.BroadcastOp(
                                self, src.handle, init, indices_a
                            )
                            return Proxy(broadcast, dst_type)
                    # do not allow cast dtype and broadcast at the same time
            # Case 4: src.type is BufferType
            if isinstance(src.type, BufferType) and isinstance(dst_type, BufferType):
                if src.type == dst_type:
                    return src

        src_ty = src.type if isinstance(src, Proxy) else "constexpr"
        self.compile_error(
            f"Cannot cast from {src_ty} to {dst_type}, unsupported type combination or value is not broadcastable"
        )

    def normalize_indices(
        self,
        indices: Sequence[Proxy | Constexpr | tuple],
        *,
        expected_len: int | None = None,
        context: str | None = None,
    ) -> list[Proxy]:
        out = []
        for val in indices:
            if isinstance(val, tuple):
                self.compile_error("Nested tuples are not supported in indices.")
            out.append(self.cast(val, index))

        if expected_len is not None and len(out) != expected_len:
            prefix = f"{context} " if context else ""
            self.compile_error(
                f"{prefix}expects {expected_len} indices, got {len(out)}."
            )
        return out

    @staticmethod
    def _ceil_log2(n: int) -> int:
        assert n >= 1
        return (n - 1).bit_length()

    def _get_promoted_dtype_nary_hls(
        self,
        op_name: str,
        dtypes: Sequence[DType],
        term_signs: Sequence[int] | None = None,
    ) -> DType:
        if len(dtypes) == 0:
            self.compile_error("Type promotion requires at least one operand")

        # HLS optimized n-ary integer rules for reduction-friendly add/sub/mul.
        if op_name in {"add", "sub"} and all(dt.is_int_signless() for dt in dtypes):
            if term_signs is None:
                signs = [1] * len(dtypes)
            else:
                if len(term_signs) != len(dtypes):
                    self.compile_error(
                        f"Type promotion for '{op_name}' expects {len(dtypes)} signs, got {len(term_signs)}"
                    )
                signs = list(term_signs)

            acc_signed = any(sign < 0 for sign in signs) or any(
                dt.is_int() for dt in dtypes
            )
            eff_widths = []
            for dt in dtypes:
                width = dt.primitive_width
                if acc_signed and dt.is_uint():
                    width += 1
                eff_widths.append(width)

            result_width = max(eff_widths) + self._ceil_log2(len(dtypes))
            return APInt(result_width, signed=acc_signed)

        if op_name == "mul" and all(dt.is_int_signless() for dt in dtypes):
            result_width = sum(dt.primitive_width for dt in dtypes)
            result_signed = any(dt.is_int() for dt in dtypes)
            return APInt(result_width, signed=result_signed)

        return self._lookup_promoted_dtype_nary(
            table=TypeTable, style_name="hls", op_name=op_name, dtypes=dtypes
        )

    def _lookup_promoted_dtype_nary(
        self,
        *,
        table,
        style_name: str,
        op_name: str,
        dtypes: Sequence[DType],
    ) -> DType:
        if len(dtypes) == 0:
            self.compile_error("Type promotion requires at least one operand")

        if len(dtypes) == 1:
            ret = table.lookup_unary(op_name, dtypes[0])
            if ret is None:
                self.compile_error(
                    f"No {style_name} type promotion rule for operator '{op_name}' with operand type {dtypes[0]}"
                )
            return ret

        ret = dtypes[0]
        for next_ty in dtypes[1:]:
            merged = table.lookup_binary(op_name, ret, next_ty)
            if merged is None:
                self.compile_error(
                    f"No {style_name} type promotion rule for operator '{op_name}' with operand types {ret} and {next_ty}"
                )
            ret = merged
        return ret

    def _get_promoted_dtype_nary_cpp(
        self,
        op_name: str,
        dtypes: Sequence[DType],
    ) -> DType:
        return self._lookup_promoted_dtype_nary(
            table=CppTypeTable, style_name="cpp", op_name=op_name, dtypes=dtypes
        )

    def get_promoted_dtype_nary(
        self,
        op_name: str,
        dtypes: Sequence[DType],
        term_signs: Sequence[int] | None = None,
    ) -> DType:
        if self.typing_style == "cpp":
            return self._get_promoted_dtype_nary_cpp(op_name, dtypes)
        return self._get_promoted_dtype_nary_hls(op_name, dtypes, term_signs)

    def get_promoted_dtype(self, lhs: DType, rhs: DType | None, op_name: str):
        if rhs is None:
            ret = self.get_promoted_dtype_nary(op_name, [lhs])
        else:
            signs = [1, -1] if op_name == "sub" else None
            ret = self.get_promoted_dtype_nary(op_name, [lhs, rhs], term_signs=signs)
        return ret

    def _reduce_balanced(
        self, operands: Sequence[Proxy], combine: Callable[[Proxy, Proxy], Proxy]
    ) -> Proxy:
        if len(operands) == 0:
            self.compile_error("Reduction requires at least one operand")
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
        self, operands: Sequence[Proxy], combine: Callable[[Proxy, Proxy], Proxy]
    ) -> Proxy:
        return self._reduce_balanced(operands, combine)

    def create_add_nary(
        self, operands: Sequence[Proxy], *, floating: bool = False
    ) -> Proxy:
        return self._reduce_balanced(
            operands, lambda lhs, rhs: self.create_add(lhs, rhs, floating=floating)
        )

    def create_sub_nary(
        self,
        operands: Sequence[Proxy],
        term_signs: Sequence[int],
        *,
        floating: bool = False,
    ) -> Proxy:
        if len(operands) != len(term_signs):
            self.compile_error(
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
        self, operands: Sequence[Proxy], *, floating: bool = False
    ) -> Proxy:
        return self._reduce_balanced(
            operands, lambda lhs, rhs: self.create_mul(lhs, rhs, floating=floating)
        )

    ######################
    # Basic arithmetic ops
    ######################
    def _create_elementwise_binary_linalg(
        self, lhs: Proxy, rhs: Proxy, build_fn: Callable
    ):
        res_ir_type = lhs.dtype.to_mlir(self.context)
        assert isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType)
        init = tensor.EmptyOp(self, lhs.type.shape, res_ir_type).get_result_at(0)
        identities = [AffineMap.get_identity(lhs.type.rank, self.context)] * 3
        iterators = [linalg.PAR] * lhs.type.rank

        # create a generic op
        op = linalg.GenericOp(
            self, [res_ir_type], [lhs.handle, rhs.handle], [init], identities, iterators
        )
        # build the region
        body = op.add_entry_block()
        lhs_arg, rhs_arg = body.get_arg_at(0), body.get_arg_at(1)
        lhs = Proxy(lhs_arg, lhs.dtype)
        rhs = Proxy(rhs_arg, rhs.dtype)
        ip = self.save_insertion_point()

        self.set_insertion_point_to_end(body)
        res = build_fn(lhs, rhs)
        linalg.YieldOp(self, [res])

        self.restore_insertion_point(ip)
        return Proxy(op.get_result_at(0), lhs.type)

    def create_add(self, lhs: Proxy, rhs: Proxy, *, floating: bool = False) -> Proxy:
        def build_fn(lhs, rhs):
            if floating:
                return arith.AddFOp(self, lhs.handle, rhs.handle).get_result_at(0)
            else:
                return arith.AddIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_sub(self, lhs: Proxy, rhs: Proxy, *, floating: bool = False) -> Proxy:
        def build_fn(lhs, rhs):
            if floating:
                return arith.SubFOp(self, lhs.handle, rhs.handle).get_result_at(0)
            else:
                return arith.SubIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_mul(self, lhs: Proxy, rhs: Proxy, *, floating: bool = False) -> Proxy:
        def build_fn(lhs, rhs):
            if floating:
                return arith.MulFOp(self, lhs.handle, rhs.handle).get_result_at(0)
            else:
                return arith.MulIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_div(
        self, lhs: Proxy, rhs: Proxy, *, signed: bool = True, floating: bool = False
    ) -> Proxy:
        assert not (signed and floating)

        def build_fn(lhs, rhs):
            if floating:
                return arith.DivFOp(self, lhs.handle, rhs.handle).get_result_at(0)
            elif signed:
                return arith.DivSIOp(self, lhs.handle, rhs.handle).get_result_at(0)
            else:
                return arith.DivUIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_floordiv(
        self, lhs: Proxy, rhs: Proxy, *, signed: bool = True, floating: bool = False
    ):
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

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_mod(
        self, lhs: Proxy, rhs: Proxy, *, signed: bool = True, floating: bool = False
    ) -> Proxy:
        assert not (signed and floating)

        def build_fn(lhs, rhs):
            if floating:
                return arith.RemFOp(self, lhs.handle, rhs.handle).get_result_at(0)
            elif signed:
                return arith.RemSIOp(self, lhs.handle, rhs.handle).get_result_at(0)
            else:
                return arith.RemUIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_pow(
        self,
        base: Proxy,
        exp: Proxy,
        *,
        base_floating: bool = False,
        exp_floating: bool = False,
    ) -> Proxy:
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

        if isinstance(base.type, ShapedType) and isinstance(exp.type, ShapedType):
            return self._create_elementwise_binary_linalg(base, exp, build_fn)
        else:
            return Proxy(build_fn(base, exp), base.dtype)

    def create_lshift(self, lhs: Proxy, rhs: Proxy) -> Proxy:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)

        def build_fn(lhs, rhs):
            return arith.ShLIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_rshift(self, lhs: Proxy, rhs: Proxy, signed: bool = True) -> Proxy:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)

        def build_fn(lhs, rhs):
            if signed:
                return arith.ShRSIOp(self, lhs.handle, rhs.handle).get_result_at(0)
            else:
                return arith.ShRUIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def _create_elementwise_unary_linalg(self, operand: Proxy, build_fn: Callable):
        res_ir_type = operand.dtype.to_mlir(self.context)
        assert isinstance(operand.type, ShapedType)
        init = tensor.EmptyOp(self, operand.type.shape, res_ir_type).get_result_at(0)
        identities = [AffineMap.get_identity(operand.type.rank, self.context)] * 2
        iterators = [linalg.PAR] * operand.type.rank

        # create a generic op
        op = linalg.GenericOp(
            self, [res_ir_type], [operand.handle], [init], identities, iterators
        )
        # build the region
        body = op.add_entry_block()
        operand_arg = body.get_arg_at(0)
        operand = Proxy(operand_arg, operand.dtype)
        ip = self.save_insertion_point()

        self.set_insertion_point_to_end(body)
        res = build_fn(operand)
        linalg.YieldOp(self, [res])

        self.restore_insertion_point(ip)
        return Proxy(op.get_result_at(0), operand.type)

    def create_neg(self, operand: Proxy, *, floating: bool = False) -> Proxy:
        def build_fn(proxy):
            if floating:
                return arith.NegFOp(self, proxy.handle).get_result_at(0)
            else:
                # for integer, neg is the same as 0 - operand
                zero = self.make_scalar(0, proxy.dtype).handle
                return arith.SubIOp(self, zero, proxy.handle).get_result_at(0)

        if isinstance(operand.type, ShapedType):
            return self._create_elementwise_unary_linalg(operand, build_fn)
        else:
            return Proxy(build_fn(operand), operand.dtype)

    def create_invert(self, operand: Proxy) -> Proxy:
        assert isinstance(operand.dtype, APInt)

        def build_fn(proxy):
            # for integer, invert is the same as -1 xor operand
            ones = 2**proxy.dtype.primitive_width - 1
            ones_val = self.make_scalar(ones, proxy.dtype).handle
            return arith.XOrIOp(self, ones_val, proxy.handle).get_result_at(0)

        if isinstance(operand.type, ShapedType):
            return self._create_elementwise_unary_linalg(operand, build_fn)
        else:
            return Proxy(build_fn(operand), operand.dtype)

    #########################
    # Comparison ops
    #########################

    def create_cmpi(
        self, lhs: Proxy, rhs: Proxy, predicate: CmpPred, *, signed=False
    ) -> Proxy:
        assert isinstance(lhs.dtype, (IndexType, APInt)) and isinstance(
            rhs.dtype, (IndexType, APInt)
        )
        pred_val = predicate.value
        if not signed and predicate in {CmpPred.LT, CmpPred.LE, CmpPred.GT, CmpPred.GE}:
            pred_val += 4  # add offset to get unsigned predicate value

        def build_fn(lhs, rhs):
            return arith.CmpIOp(self, pred_val, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            # special handle, because return type of cmpi is always int1,
            # which is not the same as lhs/rhs dtype
            handle = self._create_elementwise_binary_linalg(lhs, rhs, build_fn).handle
            if isinstance(lhs.type, TensorType):
                new_type = TensorType(int1, lhs.type.shape)
            else:
                new_type = BufferType(int1, lhs.type.shape)
            return Proxy(handle, new_type)
        else:
            return Proxy(build_fn(lhs, rhs), int1)

    _cmpf_pred_map = {
        CmpPred.EQ: 1,
        CmpPred.NE: 6,
        CmpPred.LT: 4,
        CmpPred.LE: 5,
        CmpPred.GT: 2,
        CmpPred.GE: 3,
    }

    def create_cmpf(
        self, lhs: Proxy, rhs: Proxy, predicate: CmpPred, *, ordered: bool = False
    ) -> Proxy:
        assert isinstance(lhs.dtype, APFloat) and isinstance(rhs.dtype, APFloat)
        pred_val = self._cmpf_pred_map[predicate]
        if ordered and predicate in {CmpPred.LT, CmpPred.LE, CmpPred.GT, CmpPred.GE}:
            pred_val += 7  # add offset to get ordered predicate value

        def build_fn(lhs, rhs):
            return arith.CmpFOp(self, pred_val, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            # special handle, because return type of cmpf is always int1,
            # which is not the same as lhs/rhs dtype
            handle = self._create_elementwise_binary_linalg(lhs, rhs, build_fn).handle
            if isinstance(lhs.type, TensorType):
                new_type = TensorType(int1, lhs.type.shape)
            else:
                new_type = BufferType(int1, lhs.type.shape)
            return Proxy(handle, new_type)
        else:
            return Proxy(build_fn(lhs, rhs), int1)

    def create_max(
        self,
        lhs: Proxy,
        rhs: Proxy,
        *,
        signed: bool = True,
        floating: bool = False,
        propagate_nan: bool = True,
    ):
        assert not (signed and floating)

        def build_fn(lhs, rhs):
            if floating:
                if propagate_nan:
                    return arith.MaximumFOp(self, lhs.handle, rhs.handle).get_result_at(
                        0
                    )
                else:
                    return arith.MaxNumFOp(self, lhs.handle, rhs.handle).get_result_at(
                        0
                    )
            else:
                if signed:
                    return arith.MaxSIOp(self, lhs.handle, rhs.handle).get_result_at(0)
                else:
                    return arith.MaxUIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_min(
        self,
        lhs: Proxy,
        rhs: Proxy,
        *,
        signed: bool = True,
        floating: bool = False,
        propagate_nan: bool = True,
    ):
        assert not (signed and floating)

        def build_fn(lhs, rhs):
            if floating:
                if propagate_nan:
                    return arith.MinimumFOp(self, lhs.handle, rhs.handle).get_result_at(
                        0
                    )
                else:
                    return arith.MinNumFOp(self, lhs.handle, rhs.handle).get_result_at(
                        0
                    )
            else:
                if signed:
                    return arith.MinSIOp(self, lhs.handle, rhs.handle).get_result_at(0)
                else:
                    return arith.MinUIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    ##########################
    # Bitwise logical ops
    ##########################

    def create_and(self, lhs: Proxy, rhs: Proxy) -> Proxy:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)

        def build_fn(lhs, rhs):
            return arith.AndIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_or(self, lhs: Proxy, rhs: Proxy) -> Proxy:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)

        def build_fn(lhs, rhs):
            return arith.OrIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_xor(self, lhs: Proxy, rhs: Proxy) -> Proxy:
        assert isinstance(lhs.dtype, APInt) and isinstance(rhs.dtype, APInt)

        def build_fn(lhs, rhs):
            return arith.XOrIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    ###########################
    # Logical ops
    ###########################
    def create_logical_and(self, lhs: Proxy, rhs: Proxy) -> Proxy:
        assert lhs.dtype == int1 and rhs.dtype == int1

        def build_fn(lhs, rhs):
            return arith.AndIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_logical_or(self, lhs: Proxy, rhs: Proxy) -> Proxy:
        assert lhs.dtype == int1 and rhs.dtype == int1

        def build_fn(lhs, rhs):
            return arith.OrIOp(self, lhs.handle, rhs.handle).get_result_at(0)

        if isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType):
            return self._create_elementwise_binary_linalg(lhs, rhs, build_fn)
        else:
            return Proxy(build_fn(lhs, rhs), lhs.dtype)

    def create_logical_not(self, operand: Proxy) -> Proxy:
        assert operand.dtype == int1

        def build_fn(operand):
            # logical not can be implemented as 1 xor operand
            ones = self.make_scalar(1, operand.dtype).handle
            return arith.XOrIOp(self, ones, operand.handle).get_result_at(0)

        if isinstance(operand.type, ShapedType):
            return self._create_elementwise_unary_linalg(operand, build_fn)
        else:
            return Proxy(build_fn(operand), operand.dtype)

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

    def create_broadcast(self, lhs: Proxy, rhs: Proxy) -> tuple[Proxy, Proxy]:
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
                self.compile_error(
                    f"Shapes {lhs_shape} and {rhs_shape} are not broadcastable"
                )
            init = tensor.EmptyOp(
                self, shape, lhs.dtype.to_mlir(self.context)
            ).get_result_at(0)
            if not indices_lhs and not indices_rhs:
                return lhs, rhs
            if indices_lhs:
                lhs_handle = linalg.BroadcastOp(
                    self, lhs.handle, init, indices_lhs
                ).get_result_at(0)
                if isinstance(lhs.type, TensorType):
                    lhs_type = TensorType(cast(DType, lhs.dtype), shape)
                else:
                    lhs_type = BufferType(cast(DType, lhs.dtype), shape)
                return Proxy(lhs_handle, lhs_type), rhs
            else:
                rhs_handle = linalg.BroadcastOp(
                    self, rhs.handle, init, indices_rhs
                ).get_result_at(0)
                if isinstance(rhs.type, TensorType):
                    rhs_type = TensorType(cast(DType, rhs.dtype), shape)
                else:
                    rhs_type = BufferType(cast(DType, rhs.dtype), shape)
                return lhs, Proxy(rhs_handle, rhs_type)

        if not lhs_is_shaped and not rhs_is_shaped:
            # both are scalars, no need to broadcast
            return lhs, rhs

        if isinstance(lhs.type, BufferType) or isinstance(rhs.type, BufferType):
            self.compile_error("Scalars cannot broadcast to buffer types")
        # one is scalar, it can be broadcasted to any shape
        if lhs_is_shaped and not rhs_is_shaped:
            rhs_handle = tensor.SplatOp(
                self, rhs.handle, cast(ShapedType, lhs.type).shape
            )
            return lhs, Proxy(rhs_handle, lhs.type)
        else:
            lhs_handle = tensor.SplatOp(
                self, lhs.handle, cast(ShapedType, rhs.type).shape
            )
            return Proxy(lhs_handle, rhs.type), rhs

    ###########################
    # Memory operations
    ###########################

    def make_buffer(self, type: ShapedType) -> Proxy:
        if isinstance(type, BufferType):
            alloc_op = memref.AllocOp(self, type.to_mlir(self.context))
            return Proxy(alloc_op, type)
        elif isinstance(type, TensorType):
            empty_op = tensor.EmptyOp(
                self, type.shape, type.dtype.to_mlir(self.context)
            )
            return Proxy(empty_op, type)
        else:
            self.compile_error(f"Unsupported shaped type: {type}")

    def fill_buffer(self, buffer: Proxy, value: Proxy) -> Proxy | None:
        assert isinstance(buffer.type, ShapedType)
        fill_op = linalg.FillOp(self, value.handle, buffer.handle)
        if isinstance(buffer.type, TensorType):
            return Proxy(fill_op.get_result_at(0), buffer.type)
        else:
            # memref mode has no return vals
            return None

    def create_load(self, lhs: Proxy, indices: Sequence[Proxy]) -> Proxy:
        assert isinstance(lhs.type, ShapedType)
        if isinstance(lhs.type, BufferType):
            index_values = [idx.handle for idx in indices]
            load_op = memref.LoadOp(self, lhs.handle, index_values)
            return Proxy(load_op, lhs.dtype)
        elif isinstance(lhs.type, TensorType):
            index_values = [idx.handle for idx in indices]
            load_op = tensor.ExtractOp(self, lhs.handle, index_values)
            return Proxy(load_op, lhs.dtype)
        else:
            self.compile_error(f"Unsupported shaped type: {lhs.type}")

    def create_store(
        self, value: Proxy, buffer: Proxy, indices: Sequence[Proxy]
    ) -> Proxy | None:
        assert isinstance(buffer.type, ShapedType)
        if isinstance(buffer.type, BufferType):
            index_values = [idx.handle for idx in indices]
            memref.StoreOp(self, value.handle, buffer.handle, index_values)
        elif isinstance(buffer.type, TensorType):
            index_values = [idx.handle for idx in indices]
            insert_op = tensor.InsertOp(self, value.handle, buffer.handle, index_values)
            return Proxy(insert_op, buffer.type)
        else:
            self.compile_error(f"Unsupported shaped type: {buffer.type}")

    ########################
    # Stream operations
    ########################
    def create_stream_get(self, stream: Proxy, indices: Sequence[Proxy]) -> Proxy:
        assert isinstance(stream.type, Stream)
        get = allo.StreamGetOp(self, stream.handle, [idx.handle for idx in indices])
        return Proxy(get, stream.type.base_type)

    def create_stream_put(
        self, stream: Proxy, indices: Sequence[Proxy], value: Proxy
    ) -> None:
        assert isinstance(stream.type, Stream)
        allo.StreamPutOp(
            self, stream.handle, [idx.handle for idx in indices], value.handle
        )

    def make_stream(self, stream_type: Stream) -> Proxy:
        stream = allo.StreamCreateOp(self, stream_type.to_mlir(self.context))
        return Proxy(stream, stream_type)
