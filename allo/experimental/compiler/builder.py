# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from typing import (
    Union,
    Sequence,
    Dict,
    Optional,
)

from .errors import CompilationError
from .. import core
from ..core.library import CmpPred
from ..core.rule import TypeTable
from ..core.types import (
    scalar_type,
    tensor,
    Array,
    APInt,
    APFloat,
    _is_array_of_scalar_tensor,
    _is_scalar_tensor,
)
from .._C.liballo import (
    ir,
    allo as allo_d,
    arith as arith_d,
    tensor as tensor_d,
    linalg as linalg_d,
    math as math_d,
    memref as memref_d,
)
from .. import settings


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
    module: ir.ModuleOp

    def __init__(self, context: ir.Context):
        super().__init__(context)
        self.curr_node: Optional[ast.AST] = None

    def compile_error(self, msg: str):
        raise CompilationError(self.curr_node, msg, self.src)

    def annotate(self, val: tensor, ann: core.constexpr):
        """
        Annotate a value with a string annotation.
        """
        str_attr = ir.StringAttr.get(str(ann.value), self.context)
        val.handle.set_attr("ann", str_attr)

    def get_string_attr(self, attr: str) -> ir.StringAttr:
        return ir.StringAttr.get(attr, self.context)

    def get_unit_attr(self) -> ir.UnitAttr:
        return ir.UnitAttr.get(self.context)

    def get_dict_attr(self, d: Dict[str, ir.Attribute]) -> ir.DictionaryAttr:
        return ir.DictionaryAttr.get(self.context, d)

    def get_identity_map(self, rank: int) -> ir.AffineMap:
        return ir.AffineMap.get_identity(rank, self.context)

    #####################
    # Constant Creation
    #####################
    def create_const_float(self, value: float, ty: scalar_type):
        ir_ty = ty.to_ir(self)
        return tensor(arith_d.ConstantFloatOp.create(self, ir_ty, value), ty)

    def create_const_int(self, value: int, ty: scalar_type):
        ir_ty = ty.to_ir(self)
        return tensor(arith_d.ConstantIntOp.create(self, ir_ty, value), ty)

    def create_const_index(self, value: int):
        return tensor(arith_d.ConstantIndexOp.create(self, value), core.index)

    def make_scalar(self, value, ty: scalar_type):
        """
        Create a scalar constant of the given value and type.

        :param value: the value of the scalar constant, must be a Python primitive (e.g. int, float)
        :param ty: the scalar type of the constant, must be a `scalar_type`
        :type ty: scalar_type
        :return: a `tensor` proxy value representing the scalar constant
        :rtype: tensor
        """
        if ty.is_float():
            return self.create_const_float(float(value), ty)
        elif ty.is_int_signless():
            return self.create_const_int(int(value), ty)
        elif ty.is_index():
            return self.create_const_index(int(value))
        else:
            self.compile_error(f"Unsupported scalar type: {ty}")

    def make_or_cast(self, value: core.constexpr | tensor, ty: scalar_type):
        """
        Create a constant
        """
        if isinstance(value, core.constexpr):
            return self.make_scalar(value.value, ty)
        if _is_scalar_tensor(value):
            return self.scalar_cast(value, ty)
        if _is_array_of_scalar_tensor(value):
            return self.tensor_cast(value, ty)
        assert False

    #####################
    # Type Casting
    #####################

    def _create_index_cast_op(self, val: ir.Value, dst_ty: ir.Type):
        return arith_d.IndexCastOp.create(self, dst_ty, val)

    def _create_ext_op(
        self, val: ir.Value, target_ty: ir.Type, signed: bool = True, floating=False
    ) -> ir.Value:
        if floating:
            return arith_d.ExtFOp.create(self, val, target_ty)
        else:
            if signed:
                return arith_d.ExtSIOp.create(self, val, target_ty)
            else:
                return arith_d.ExtUIOp.create(self, val, target_ty)

    def _create_trunc_op(
        self, val: ir.Value, target_ty: ir.Type, floating=False
    ) -> ir.Value:
        if floating:
            return arith_d.TruncFOp.create(self, val, target_ty)
        else:
            return arith_d.TruncIOp.create(self, val, target_ty)

    def _create_itofp_op(
        self, val: ir.Value, target_ty: ir.Type, signed: bool = True
    ) -> ir.Value:
        if signed:
            return arith_d.SIToFPOp.create(self, val, target_ty)
        else:
            return arith_d.UIToFPOp.create(self, val, target_ty)

    def _create_fptoi_op(
        self, val: ir.Value, target_ty: ir.Type, signed: bool = True
    ) -> ir.Value:
        if signed:
            return arith_d.FPToSIOp.create(self, val, target_ty)
        else:
            return arith_d.FPToUIOp.create(self, val, target_ty)

    def scalar_cast(self, src: tensor, dst_ty: scalar_type):
        """
        Perform scalar type casting, user must ensure that `src` is a scalar and `dst_ty` is a scalar type. The method will return a new `tensor` proxy value with the same value as `src` but with type `dst_ty`.

        If `src` is already of type `dst_ty`, it will return `src` directly.

        :param src: the source scalar value to be casted, must be a `tensor` proxy value with scalar type
        :type src: tensor
        :param dst_ty: the target scalar type to cast to, must be a `scalar_type`
        :type dst_ty: scalar_type
        :return: a new `tensor` proxy value with the same value as `src` but with type `dst_ty`
        :rtype: tensor
        """
        assert isinstance(src.type, scalar_type)
        src_ty: scalar_type = src.type
        val = src.handle
        if src_ty == dst_ty:
            return src
        if src_ty.is_int() and dst_ty.is_index():
            index_ty = dst_ty.to_ir(self)
            return tensor(self._create_index_cast_op(val, index_ty), dst_ty)
        # int to int
        if src_ty.is_int() and dst_ty.is_int():
            int_ty = dst_ty.to_ir(self)
            if src_ty.primitive_width < dst_ty.primitive_width:
                return tensor(self._create_ext_op(val, int_ty), dst_ty)
            else:
                return tensor(self._create_trunc_op(val, int_ty), dst_ty)
        # int to uint
        if src_ty.is_int() and dst_ty.is_uint():
            int_ty = dst_ty.to_ir(self)
            if src_ty.primitive_width < dst_ty.primitive_width:
                return tensor(self._create_ext_op(val, int_ty, signed=False), dst_ty)
            else:
                if src_ty.primitive_width == dst_ty.primitive_width:
                    # if they have the same width, we can just reinterpret cast
                    return tensor(val, dst_ty)
                return tensor(self._create_trunc_op(val, int_ty), dst_ty)
        # uint to index
        if src_ty.is_uint() and dst_ty.is_index():
            index_ty = dst_ty.to_ir(self)
            return tensor(self._create_index_cast_op(val, index_ty), dst_ty)
        # uint to int
        if src_ty.is_uint() and dst_ty.is_int():
            int_ty = dst_ty.to_ir(self)
            if src_ty.primitive_width < dst_ty.primitive_width:
                return tensor(self._create_ext_op(val, int_ty), dst_ty)
            else:
                if src_ty.primitive_width == dst_ty.primitive_width:
                    # if they have the same width, we can just reinterpret cast
                    return tensor(val, dst_ty)
                return tensor(self._create_trunc_op(val, int_ty), dst_ty)
        # uint to uint
        if src_ty.is_uint() and dst_ty.is_uint():
            int_ty = dst_ty.to_ir(self)
            if src_ty.primitive_width < dst_ty.primitive_width:
                return tensor(self._create_ext_op(val, int_ty, signed=False), dst_ty)
            else:
                return tensor(self._create_trunc_op(val, int_ty), dst_ty)
        # int/uint to float
        if src_ty.is_int_signless() and dst_ty.is_float():
            signed = src_ty.is_int()
            fp_ty = dst_ty.to_ir(self)
            return tensor(self._create_itofp_op(val, fp_ty, signed=signed), dst_ty)
        # float to int/uint
        if src_ty.is_float() and dst_ty.is_int_signless():
            signed = dst_ty.is_int()
            int_ty = dst_ty.to_ir(self)
            return tensor(self._create_fptoi_op(val, int_ty, signed=signed), dst_ty)
        # float to float
        if src_ty.is_float() and dst_ty.is_float():
            fp_ty = dst_ty.to_ir(self)
            if src_ty.primitive_width < dst_ty.primitive_width:
                return tensor(self._create_ext_op(val, fp_ty, floating=True), dst_ty)
            else:
                return tensor(self._create_trunc_op(val, fp_ty, floating=True), dst_ty)
        # index to int
        if src_ty.is_index() and dst_ty.is_int():
            int_ty = dst_ty.to_ir(self)
            return tensor(self._create_index_cast_op(val, int_ty), dst_ty)
        # index to uint
        if src_ty.is_index() and dst_ty.is_uint():
            int_ty = dst_ty.to_ir(self)
            return tensor(self._create_index_cast_op(val, int_ty), dst_ty)
        # index to float
        if src_ty.is_index() and dst_ty.is_float():
            # first convert index to int
            int_ty = core.APInt(src_ty.primitive_width).to_ir(self)
            int_val = self._create_index_cast_op(val, int_ty)
            fp_ty = dst_ty.to_ir(self)
            return tensor(self._create_itofp_op(int_val, fp_ty, signed=True), dst_ty)
        self.compile_error(f"Unsupported scalar cast: {src_ty} to {dst_ty}")

    def tensor_cast(self, src: tensor, dst_dtype: scalar_type):
        """
        Perform tensor type casting. It creates a new tensor with the same shape as `src` but with data type `dst_dtype`.

        :param src: the source tensor to be casted, must be a `tensor` proxy value
        :type src: tensor
        :param dst_dtype: the target data type to cast to, must be a `scalar_type`
        :type dst_dtype: scalar_type
        :return: a new `tensor` proxy value with the same shape as `src` but with data type `dst_dtype`
        :rtype: tensor
        """
        src_ty = src.dtype
        val = src.handle
        if src_ty == dst_dtype:
            return src
        # create a new tensor type with target dtype
        dst_ty = core.Array(dst_dtype, src.shape)
        ret = tensor_d.CastOp.create(self, val, dst_ty.to_ir(self))
        return tensor(ret, dst_ty)

    def cast(self, src: tensor, dst_ty: scalar_type):
        """
        Perform type casting, it will automatically determine whether to perform scalar cast or tensor cast based on the type of `src`.

        If `src` is a scalar tensor (i.e. a tensor with empty shape), it will perform scalar cast and return a scalar tensor with the same shape but with type `dst_ty`.

        If `src` is a non-scalar tensor, it will perform tensor cast and return a new tensor with the same shape as `src` but with data type `dst_ty`.

        :param src: the source value to be casted, must be a `tensor` proxy value
        :type src: tensor
        :param dst_ty: the target scalar type to cast to, must be a `scalar_type`
        :type dst_ty: scalar_type
        :return: a new `tensor` proxy value representing the casted result
        :rtype: tensor
        """
        if isinstance(src.type, scalar_type):
            return self.scalar_cast(src, dst_ty)
        elif isinstance(src.type, core.Array):
            return self.tensor_cast(src, dst_ty)
        else:
            self.compile_error(f"Unsupported type for cast: {src.type}")

    def get_promoted_dtype(
        self, lhs_ty: scalar_type, rhs_ty: Union[scalar_type, None], op_name: str
    ):
        """
        Lookup the Allo typing rules to get the promoted data type for the given operation and operand types.

        See `core/rule.py` for the typing rules.

        :param lhs_ty: left hand side scalar type
        :param rhs_ty: right hand side scalar type, or None for unary operations
        :param op_name: the name of the operation, e.g. "add", "sub", "mul", etc.
        """
        if rhs_ty is None:
            ret = TypeTable.lookup_unary(op_name, lhs_ty)
        else:
            ret = TypeTable.lookup_binary(op_name, lhs_ty, rhs_ty)
        if ret is None:
            self.compile_error(
                f"Unsupported operand type(s) for {op_name}: '{lhs_ty}' and '{rhs_ty}'"
            )
        return ret

    ######################
    # Basic arithmetic ops
    ######################

    def _create_elementwise_linalg_op(self, lhs: tensor, rhs: tensor, build_fn):
        res_ir_ty = lhs.type.to_ir(self)
        init = tensor_d.EmptyOp.create(self, res_ir_ty)
        identities = [ir.AffineMap.get_identity(len(lhs.shape), self.context)] * 3
        iterators = [linalg_d.PAR] * len(lhs.shape)
        # create a generic op
        op = linalg_d.GenericOp.create(
            self, [res_ir_ty], [lhs.handle, rhs.handle], [init], identities, iterators
        )
        # build body
        body = op.add_entry_block()
        lhs_arg, rhs_arg = body.get_arg_at(0), body.get_arg_at(1)
        self.set_insertion_point_to_start(body)
        res = build_fn(lhs_arg, rhs_arg)
        # must end with linalg.yield
        linalg_d.YieldOp.create(self, [res])
        return tensor(op.get_result_at(0), lhs.type)

    def create_add(
        self, lhs: tensor, rhs: tensor, use_linalg: bool = settings.USE_TENSOR
    ):
        """
        Create addition op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param use_linalg: whether to use `linalg.add` op
        """
        floating = isinstance(lhs.dtype, APFloat)

        def arith_add(lhs, rhs):
            if floating:
                return arith_d.AddFOp.create(self, lhs, rhs)
            else:
                return arith_d.AddIOp.create(self, lhs, rhs)

        if use_linalg and isinstance(lhs.type, Array):
            return self._create_elementwise_linalg_op(lhs, rhs, arith_add)
        else:
            ret = arith_add(lhs.handle, rhs.handle)
            return tensor(ret, lhs.type)

    def create_sub(
        self, lhs: tensor, rhs: tensor, use_linalg: bool = settings.USE_TENSOR
    ):
        """
        Create subtraction op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param use_linalg: whether to use `linalg.sub` op
        """
        floating = isinstance(lhs.dtype, APFloat)

        def arith_sub(lhs, rhs):
            if floating:
                return arith_d.SubFOp.create(self, lhs, rhs)
            else:
                return arith_d.SubIOp.create(self, lhs, rhs)

        if use_linalg and isinstance(lhs.type, Array):
            return self._create_elementwise_linalg_op(lhs, rhs, arith_sub)
        else:
            ret = arith_sub(lhs.handle, rhs.handle)
            return tensor(ret, lhs.type)

    def create_mul(
        self, lhs: tensor, rhs: tensor, use_linalg: bool = settings.USE_TENSOR
    ):
        """
        Create multiplication op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param use_linalg: whether to use `linalg.mul` op
        """
        floating = isinstance(lhs.dtype, APFloat)

        def arith_mul(lhs, rhs):
            if floating:
                return arith_d.MulFOp.create(self, lhs, rhs)
            else:
                return arith_d.MulIOp.create(self, lhs, rhs)

        if use_linalg and isinstance(lhs.type, Array):
            return self._create_elementwise_linalg_op(lhs, rhs, arith_mul)
        else:
            ret = arith_mul(lhs.handle, rhs.handle)
            return tensor(ret, lhs.type)

    def create_div(
        self,
        lhs: tensor,
        rhs: tensor,
        signed: bool = True,
        use_linalg: bool = settings.USE_TENSOR,
    ):
        """
        Create division op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param signed: whether the division is signed (only applicable for integer types)
        :param use_linalg: whether to use `linalg.div` op
        """
        floating = isinstance(lhs.dtype, APFloat)

        def arith_div(lhs, rhs):
            if floating:
                return arith_d.DivFOp.create(self, lhs, rhs)
            if signed:
                return arith_d.DivSIOp.create(self, lhs, rhs)
            else:
                return arith_d.DivUIOp.create(self, lhs, rhs)

        if use_linalg and isinstance(lhs.type, Array):
            return self._create_elementwise_linalg_op(lhs, rhs, arith_div)
        else:
            ret = arith_div(lhs.handle, rhs.handle)
            return tensor(ret, lhs.type)

    def create_floordiv(
        self,
        lhs: tensor,
        rhs: tensor,
        signed: bool = True,
        use_linalg: bool = settings.USE_TENSOR,
    ):
        """
        Create floor division op, user must ensure that a and b are of the same type.

        Broadcasting should have been handled by the caller.

        :param signed: whether the division is signed (only applicable for integer types)
        :param use_linalg: whether to use `linalg.floordiv` op
        """
        floating = isinstance(lhs.dtype, APFloat)

        def arith_floordiv(lhs, rhs):
            if floating:
                divf = arith_d.DivFOp.create(self, lhs, rhs)
                return math_d.FloorOp.create(self, divf)
            else:
                if signed:
                    return arith_d.FloorDivSIOp.create(self, lhs, rhs)
                else:
                    # for unsigned, floordivui is equivalent to divui
                    return arith_d.DivUIOp.create(self, lhs, rhs)

        if use_linalg and isinstance(lhs.type, Array):
            return self._create_elementwise_linalg_op(lhs, rhs, arith_floordiv)
        else:
            ret = arith_floordiv(lhs.handle, rhs.handle)
            return tensor(ret, lhs.type)

    def create_mod(self, lhs: tensor, rhs: tensor, signed: bool = True):
        """
        Create modulus op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param signed: whether the modulus is signed (only applicable for integer types)
        """
        if isinstance(lhs.dtype, APFloat):
            ret = arith_d.RemFOp.create(self, lhs.handle, rhs.handle)
        else:
            if signed:
                ret = arith_d.RemSIOp.create(self, lhs.handle, rhs.handle)
            else:
                ret = arith_d.RemUIOp.create(self, lhs.handle, rhs.handle)
        return tensor(ret, lhs.type)

    def create_pow(
        self, base: tensor, exponent: tensor, use_linalg: bool = settings.USE_TENSOR
    ):
        """
        Create power op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        Inputs must be floating point types.

        :param use_linalg: whether to use `linalg.pow` op
        """
        assert isinstance(base.dtype, APFloat) and isinstance(exponent.dtype, APFloat)

        def build_pow(lhs, rhs):
            return math_d.PowFOp.create(self, lhs, rhs)

        if use_linalg and isinstance(base.type, Array):
            return self._create_elementwise_linalg_op(base, exponent, build_pow)
        else:
            ret = build_pow(base.handle, exponent.handle)
            return tensor(ret, base.type)

    def create_lshift(self, lhs: tensor, rhs: tensor):
        """
        Create left shift op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param use_linalg: whether to use `linalg.lshift` op
        """
        assert isinstance(lhs.type, APInt) and isinstance(rhs.type, APInt)
        return tensor(arith_d.ShLIOp.create(self, lhs.handle, rhs.handle), lhs.type)

    def create_rshift(self, lhs: tensor, rhs: tensor, signed: bool = True):
        """
        Create right shift op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param signed: whether the right shift is signed (only applicable for integer types)
        :param use_linalg: whether to use `linalg.rshift` op
        """
        assert isinstance(lhs.type, APInt) and isinstance(rhs.type, APInt)
        if signed:
            ret = arith_d.ShRSIOp.create(self, lhs.handle, rhs.handle)
        else:
            ret = arith_d.ShRUIOp.create(self, lhs.handle, rhs.handle)
        return tensor(ret, lhs.type)

    def create_neg(self, operand: tensor, use_linalg: bool = settings.USE_TENSOR):
        """
        Create negation op, user must ensure that operand is of a numeric type

        :param use_linalg: whether to use `linalg.neg` op
        """
        if use_linalg and isinstance(operand.type, Array):
            zero = self.make_scalar(0, operand.dtype)
            init = tensor_d.EmptyOp.create(self, operand.type.to_ir(self))
            zeros = linalg_d.FillOp.create(self, zero.handle, init).get_result_at(0)
            ret = linalg_d.SubOp.create(
                self, zeros, operand.handle, init
            ).get_result_at(0)
        else:
            if isinstance(operand.dtype, APFloat):
                ret = arith_d.NegFOp.create(self, operand.handle)
            else:
                zero = self.make_scalar(0, operand.dtype)
                ret = arith_d.SubIOp.create(self, zero.handle, operand.handle)
        return tensor(ret, operand.type)

    def create_invert(self, operand: tensor):
        """
        Create bitwise invert op, user must ensure that operand is of an integer type

        :param operand: the input tensor to be inverted, must be a `tensor` proxy value with integer data type
        :return: a new `tensor` proxy value representing the bitwise invert of the input tensor
        :rtype: tensor
        """
        assert isinstance(operand.dtype, APInt)
        ones = 2**operand.dtype.primitive_width - 1
        all_ones = self.make_scalar(ones, operand.dtype).handle
        if isinstance(operand.type, Array):
            init = tensor_d.EmptyOp.create(self, operand.type.to_ir(self))
            all_ones_tensor = linalg_d.FillOp.create(
                self, all_ones, init
            ).get_result_at(0)
            ret = arith_d.XOrIOp.create(self, operand.handle, all_ones_tensor)
        else:
            ret = arith_d.XOrIOp.create(self, operand.handle, all_ones)
        return tensor(ret, operand.type)

    #########################
    # Comparison ops
    #########################
    def create_cmpi(self, lhs: tensor, rhs: tensor, pred: CmpPred, signed=False):
        """
        Create integer comparison op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param pred: the comparison predicate, must be a `CmpI` enum value
        """
        pred_val = pred.value
        if signed and pred in {CmpPred.LT, CmpPred.LE, CmpPred.GT, CmpPred.GE}:
            pred_val += 4  # add an offset to convert to signed predicate
        ret = arith_d.CmpIOp.create(self, pred_val, lhs.handle, rhs.handle)
        if isinstance(lhs.type, Array):
            ret_ty = core.Array(core.int1, lhs.shape)
        else:
            ret_ty = core.int1
        return tensor(ret, ret_ty)

    _cmpf_pred_map = {
        CmpPred.EQ: 1,
        CmpPred.NE: 6,
        CmpPred.LT: 4,
        CmpPred.LE: 5,
        CmpPred.GT: 2,
        CmpPred.GE: 3,
    }

    def create_cmpf(self, lhs: tensor, rhs: tensor, pred: CmpPred, ordered=False):
        """
        Create floating point comparison op, user must ensure that a and b are of the same type

        Broadcasting should have been handled by the caller.

        :param pred: the comparison predicate, must be a `CmpF` enum value
        :param ordered: whether to use ordered comparison (only applicable for GT, GE, LT, LE predicates)
        """
        pred_val = self._cmpf_pred_map[pred]
        if ordered and pred in {CmpPred.LT, CmpPred.LE, CmpPred.GT, CmpPred.GE}:
            pred_val += 7  # add an offset to convert to ordered predicate
        ret = arith_d.CmpFOp.create(self, pred_val, lhs.handle, rhs.handle)
        if isinstance(lhs.type, Array):
            ret_ty = core.Array(core.int1, lhs.shape)
        else:
            ret_ty = core.int1
        return tensor(ret, ret_ty)

    def create_max(
        self,
        lhs: tensor,
        rhs: tensor,
        signed=True,
        propagate_nan=True,
        use_linalg: bool = settings.USE_TENSOR,
    ):
        """
        Create integer max op, user must ensure that a and b are of the same type.

        Broadcasting should have been handled by the caller.

        :param signed: whether the max is signed (only applicable for integer types)
        """
        floating = isinstance(lhs.dtype, APFloat)

        def build_max(lhs, rhs):
            if floating:
                if propagate_nan:
                    return arith_d.MaximumFOp.create(self, lhs, rhs)
                else:
                    return arith_d.MaxNumFOp.create(self, lhs, rhs)
            else:
                if signed:
                    return arith_d.MaxSIOp.create(self, lhs, rhs)
                else:
                    return arith_d.MaxUIOp.create(self, lhs, rhs)

        if use_linalg and isinstance(lhs.type, Array):
            return self._create_elementwise_linalg_op(lhs, rhs, build_max)
        else:
            ret = build_max(lhs.handle, rhs.handle)
            return tensor(ret, lhs.type)

    def create_min(
        self,
        lhs: tensor,
        rhs: tensor,
        signed=True,
        propagate_nans=True,
        use_linalg: bool = settings.USE_TENSOR,
    ):
        """
        Create integer min op, user must ensure that a and b are of the same type.

        Broadcasting should have been handled by the caller.

        :param signed: whether the min is signed (only applicable for integer types)
        """
        floating = isinstance(lhs.dtype, APFloat)

        def build_min(lhs, rhs):
            if floating:
                if propagate_nans:
                    return arith_d.MinimumFOp.create(self, lhs, rhs)
                else:
                    return arith_d.MinNumFOp.create(self, lhs, rhs)
            else:
                if signed:
                    return arith_d.MinSIOp.create(self, lhs, rhs)
                else:
                    return arith_d.MinUIOp.create(self, lhs, rhs)

        if use_linalg and isinstance(lhs.type, Array):
            return self._create_elementwise_linalg_op(lhs, rhs, build_min)
        else:
            ret = build_min(lhs.handle, rhs.handle)
            return tensor(ret, lhs.type)

    ##########################
    # Bitwise logical ops
    ##########################

    def create_bitwise_and(self, lhs: tensor, rhs: tensor):
        """
        Create bitwise and op, user must ensure that a and b are of the same integer type

        Broadcasting should have been handled by the caller.
        """
        assert isinstance(lhs.type, APInt) and isinstance(rhs.type, APInt)
        return tensor(arith_d.AndIOp.create(self, lhs.handle, rhs.handle), lhs.type)

    def create_bitwise_or(self, lhs: tensor, rhs: tensor):
        """
        Create bitwise or op, user must ensure that a and b are of the same integer type

        Broadcasting should have been handled by the caller.
        """
        assert isinstance(lhs.type, APInt) and isinstance(rhs.type, APInt)
        return tensor(arith_d.OrIOp.create(self, lhs.handle, rhs.handle), lhs.type)

    def create_bitwise_xor(self, lhs: tensor, rhs: tensor):
        """
        Create bitwise xor op, user must ensure that a and b are of the same integer type

        Broadcasting should have been handled by the caller.
        """
        assert isinstance(lhs.type, APInt) and isinstance(rhs.type, APInt)
        return tensor(arith_d.XOrIOp.create(self, lhs.handle, rhs.handle), lhs.type)

    ###########################
    # Logical ops
    ###########################
    def create_logical_and(self, lhs: tensor, rhs: tensor):
        """
        Create logical and op, user must ensure that a and b are of the same boolean type

        Broadcasting should have been handled by the caller.
        """
        assert lhs.dtype == core.int1 and rhs.dtype == core.int1
        return tensor(arith_d.AndIOp.create(self, lhs.handle, rhs.handle), core.int1)

    def create_logical_or(self, lhs: tensor, rhs: tensor):
        """
        Create logical or op, user must ensure that a and b are of the same boolean type

        Broadcasting should have been handled by the caller.
        """
        assert lhs.dtype == core.int1 and rhs.dtype == core.int1
        return tensor(arith_d.OrIOp.create(self, lhs.handle, rhs.handle), core.int1)

    def create_logical_not(self, operand: tensor):
        """
        Create logical not op, user must ensure that operand is of boolean type
        """
        assert operand.dtype == core.int1
        one = self.make_scalar(1, core.int1).handle
        return tensor(arith_d.XOrIOp.create(self, operand.handle, one), core.int1)

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

    def create_broadcast(
        self, lhs: tensor, rhs: tensor, use_linalg: bool = settings.USE_TENSOR
    ):
        """
        Create broadcast ops for two tensors, user must ensure that a and b are of the same
        primitive type.

        Raise compilation error if the two tensors are not broadcastable.

        :param use_linalg: whether to use `linalg.broadcast` op
        :param lhs: left hand side proxy value
        :param rhs: right hand side proxy value
        """
        lhs_is_array = isinstance(lhs.type, Array)
        rhs_is_array = isinstance(rhs.type, Array)
        if not lhs_is_array and not rhs_is_array:
            # no need to broadcast if both are scalars
            return lhs, rhs
        if not use_linalg:
            self.compile_error("Broadcasting is only supported in linalg path")
        if lhs_is_array and rhs_is_array:
            shape, indices_a, indices_b = self.infer_broadcast_shape(
                lhs.shape, rhs.shape
            )
            if not shape:
                self.compile_error(
                    f"Shapes {lhs.shape} and {rhs.shape} are not broadcastable"
                )
            if not indices_a and not indices_b:
                # no need to broadcast if they are already of the same shape
                return lhs, rhs
            if indices_a:
                lhs_hdl = linalg_d.BroadcastOp.create(
                    self, lhs.handle, rhs.handle, indices_a
                ).get_result_at(0)
                return tensor(lhs_hdl, core.Array(lhs.dtype, shape)), rhs
            else:
                rhs_hdl = linalg_d.BroadcastOp.create(
                    self, rhs.handle, lhs.handle, indices_b
                ).get_result_at(0)
                return lhs, tensor(rhs_hdl, core.Array(rhs.dtype, shape))
        elif lhs_is_array:
            # rhs is scalar, scalar can be broadcasted to any shape
            rhs_hdl = tensor_d.SplatOp.create(self, rhs.handle, lhs.shape)
            return lhs, tensor(rhs_hdl, core.Array(rhs.dtype, lhs.shape))
        else:
            # lhs is scalar, scalar can be broadcasted to any shape
            lhs_hdl = tensor_d.SplatOp.create(self, lhs.handle, rhs.shape)
            return tensor(lhs_hdl, core.Array(lhs.dtype, rhs.shape)), rhs

    #####################
    # Memory ops
    #####################
    def create_load(
        self,
        buffer: tensor,
        indices: Sequence[tensor],
        use_tensor: bool = settings.USE_TENSOR,
    ) -> tensor:
        """
        Create a load op to load a value from a buffer at the given indices.

        The operation itself does not perform any checking, and users must ensure that the buffer is an array
        and the indices are of index type and within bounds.
        """
        assert isinstance(buffer.type, Array)
        ir_hdls = [idx.handle for idx in indices]
        if use_tensor:
            ret_hdl = tensor_d.ExtractOp.create(self, buffer.handle, ir_hdls)
        else:
            ret_hdl = memref_d.LoadOp.create(
                self, buffer.handle, ir_hdls
            ).get_result_at(0)
        return tensor(ret_hdl, buffer.dtype)

    def create_store(
        self,
        buffer: tensor,
        value: tensor,
        indices: Sequence[tensor],
        use_tensor: bool = settings.USE_TENSOR,
    ) -> Optional[tensor]:
        """
        Create a store op to store a value to a buffer at the given indices. If `use_tensor` is True, create a `tensor.insert` op instead of `memref.store`, and return the new tensor with the value inserted.

        The operation itself does not perform any checking, and users must ensure that the buffer is an array,
        the value is of the same type as the buffer element type, and the indices are of index type and within bounds.
        """
        assert isinstance(buffer.type, Array)
        ir_hdls = [idx.handle for idx in indices]
        if use_tensor:
            ret_hdl = tensor_d.InsertOp.create(
                self, value.handle, buffer.handle, ir_hdls
            )
            return tensor(ret_hdl, buffer.type)
        else:
            memref_d.StoreOp.create(self, value.handle, buffer.handle, ir_hdls)
            return None

    def make_array(
        self,
        dtype: scalar_type,
        shape: Sequence[int],
        init: Optional[tensor | core.constexpr] = None,
        use_tensor: bool = settings.USE_TENSOR,
    ) -> tensor:
        """
        Create a new array with the given shape and data type, and optionally initialize it with a given value.

        The method does not perform any checking, and users must ensure that the shape is a sequence of integers, the data type is a scalar type, and the initial value (if provided) is compatible with the data type.

        :param shape: the shape of the array to be created, must be a sequence of integers
        :param dtype: the data type of the array elements, must be a `scalar_type`
        :param init: an optional initial value to fill the array with, can be either a `tensor` proxy value or a `core.constexpr` value
        :return: a `tensor` proxy value representing the created array
        :rtype: tensor
        """
        arr_ty = core.Array(dtype, shape)
        arr_ir_ty = arr_ty.to_ir(self)
        if use_tensor:
            arr = tensor_d.EmptyOp.create(self, arr_ir_ty)
        else:
            arr = memref_d.AllocOp.create(self, arr_ir_ty)
        if init is None:
            return tensor(arr, arr_ty)
        if isinstance(init, core.constexpr):
            init_value = self.make_scalar(init.value, dtype)
        else:
            assert isinstance(init, tensor) and init.dtype == dtype
            init_value = init
        fill_op = linalg_d.FillOp.create(self, init_value.handle, arr)
        if use_tensor:
            return tensor(fill_op.get_result_at(0), arr_ty)
        return tensor(arr, arr_ty)

    def fill_array(
        self,
        array: tensor,
        value: core.constexpr | tensor,
        use_tensor: bool = settings.USE_TENSOR,
    ):
        """
        Fill an existing array with a given value. The method does not perform any checking.
        """
        assert isinstance(array.type, Array)
        dtype = array.dtype
        val = self.make_or_cast(value, dtype)
        fill_op = linalg_d.FillOp.create(self, val.handle, array.handle)
        if use_tensor:
            return tensor(fill_op.get_result_at(0), array.type)
        return None

    ######################
    # SPMD related ops
    ######################
    def create_get_pid_op(self, dim: int) -> tensor:
        """
        Create an op to get the program id in the given dimension.

        :param dim: the dimension to get the program id, must be a non-negative integer
        :return: a `tensor` proxy value representing the program id in the given dimension, with scalar type `index`
        :rtype: tensor
        """
        pid_hdl = allo_d.GetProgramIdOp.create(self, dim)
        return tensor(pid_hdl, core.index)

    def create_get_n_progs_op(self, dim: int) -> tensor:
        """
        Create an op to get the number of programs in the given dimension.

        :param dim: the dimension to get the number of programs, must be a non-negative integer
        :return: a `tensor` proxy value representing the number of programs in the given dimension, with scalar type `index`
        :rtype: tensor
        """
        nprogs_hdl = allo_d.GetNumProgramsOp.create(self, dim)
        return tensor(nprogs_hdl, core.index)

    ########################
    # Channel related ops
    ########################
    def make_channel(
        self,
        name: str,
        data_ty: scalar_type | Array,
        capacity: int,
        shape: Sequence[int],
    ):
        """
        Create a new channel with the given name, data type, capacity, and shape.

        :param name: the name of the channel, must be a string
        :param data_ty: the data type of the channel elements, must be a `scalar_type` or an `Array` type
        :param capacity: the capacity of the channel, must be a non-negative integer
        :param shape: the shape of the channel elements if `data_ty` is a scalar type, must be a sequence of integers; should be empty if `data_ty` is an `Array` type
        """
        assert isinstance(data_ty, (scalar_type, Array))
        channel_ty = allo_d.ChannelType.get(self.context, data_ty.to_ir(self), capacity)
        # ret = allo_d.ChanCreateOp.create(self, name, channel_ty, shape)
        # ret_ty = core.Channel(data_ty, shape, capacity)
        # return tensor(ret, ret_ty)
        self.set_insertion_point_to_start(self.module.body)
        allo_d.ChanCreateOp.create(self, name, channel_ty, shape)
        ret_ty = core.Channel(name, data_ty, shape, capacity)
        return tensor(None, ret_ty)

    ############################
    # Bit manipulation ops
    ############################
    def create_bit_extract_op(self, src: tensor, lo: tensor, width: int) -> tensor:
        """
        Create an op to extract a bit field from an integer value.

        :param src: the source integer tensor to extract bits from, must be a `tensor` proxy value with integer data type
        :param lo: the starting bit position to extract, must be a `tensor` proxy value with scalar type `index`
        :param width: the number of bits to extract, must be a positive integer
        :return: a `tensor` proxy value representing the extracted bit field, with integer data type and the same signedness as the source type
        :rtype: tensor
        """
        assert isinstance(src.dtype, APInt)
        assert lo.dtype == core.index
        ret = allo_d.BitExtractOp.create(self, src.handle, lo.handle, width)
        # extract a unsigned bit field
        ret_ty = core.APInt(width, signed=False)
        return tensor(ret, ret_ty)

    def create_bit_insert_op(
        self, dst: tensor, src: tensor, lo: tensor, width: int
    ) -> tensor:
        """
        Create an op to insert a bit field into an integer value.

        :param dst: the destination integer tensor to insert bits into, must be a `tensor` proxy value with integer data type
        :param src: the source integer tensor to extract bits from, must be a `tensor` proxy value with integer data type and the same signedness as `dst`
        :param lo: the starting bit position to insert, must be a `tensor` proxy value with scalar type `index`
        :param width: the number of bits to insert, must be a positive integer
        :return: a `tensor` proxy value representing the result of inserting the bit field, with the same type as `dst`
        :rtype: tensor
        """
        assert isinstance(dst.dtype, APInt)
        assert isinstance(src.dtype, APInt)
        assert lo.dtype == core.index
        ret = allo_d.BitInsertOp.create(self, dst.handle, src.handle, lo.handle, width)
        return tensor(ret, dst.dtype)

    def create_bit_cast_op(self, src: tensor, dst_ty: scalar_type) -> tensor:
        """
        Create an op to perform bit cast between two scalar types of the same bit width.

        :param src: the source tensor to be bit casted, must be a `tensor` proxy value with scalar type
        :param dst_ty: the target scalar type to bit cast to, must have the same bit width as `src`'s type
        :return: a new `tensor` proxy value representing the result of the bit cast, with the same shape as `src` and data type `dst_ty`
        :rtype: tensor
        """
        assert isinstance(src.type, scalar_type)
        assert isinstance(dst_ty, scalar_type)
        if src.type.primitive_width != dst_ty.primitive_width:
            self.compile_error(
                f"Cannot bit cast from {src.type} to {dst_ty} due to different bit widths"
            )
        ret = arith_d.BitcastOp.create(self, src.handle, dst_ty.to_ir(self))
        return tensor(ret, dst_ty)
