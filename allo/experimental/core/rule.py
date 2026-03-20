# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
from .types import APInt, APFloat, fp32, fp64, DType, IndexType, index, int1


# temporary int/uint class to distinguish between signed and unsigned integers
# for typing rules
class int:
    pass


class uint:
    pass


class TypingRule:
    def __init__(self, rule_dicts, commutative=False):
        self.rules = {}
        self.commutative = commutative
        for d in rule_dicts:
            # flatten the dicts
            self.rules.update(d)

    def call_binary(self, t1, t2) -> DType | None:
        # convert signless int to signed/unsigned int
        ty_1 = type(t1) if not t1.is_int_signless() else (int if t1.is_int() else uint)
        ty_2 = type(t2) if not t2.is_int_signless() else (int if t2.is_int() else uint)
        key = (ty_1, ty_2)
        # find rule
        if key in self.rules:
            return self.rules[key](t1, t2)
        if self.commutative:
            # try swapped order
            key = (ty_2, ty_1)
            if key in self.rules:
                return self.rules[key](t2, t1)
        return None

    def call_unary(self, t) -> DType | None:
        ty = type(t) if not t.is_int_signless() else (int if t.is_int() else uint)
        key = (ty,)
        if key in self.rules:
            return self.rules[key](t)
        return None


class TypeTable:
    _REGISTRY = {}

    @classmethod
    def register(cls, op_keys: List[str], rule_obj):
        if not isinstance(op_keys, (list, tuple)):
            op_keys = [op_keys]
        for k in op_keys:
            cls._REGISTRY[k] = rule_obj

    @classmethod
    def lookup_binary(cls, op_key: str, t1, t2) -> DType | None:
        if op_key not in cls._REGISTRY:
            return None
        return cls._REGISTRY[op_key].call_binary(t1, t2)

    @classmethod
    def lookup_unary(cls, op_key: str, t) -> DType | None:
        if op_key not in cls._REGISTRY:
            return None
        return cls._REGISTRY[op_key].call_unary(t)


class CppTypeTable:
    _REGISTRY = {}

    @classmethod
    def register(cls, op_keys: List[str], rule_obj):
        if not isinstance(op_keys, (list, tuple)):
            op_keys = [op_keys]
        for k in op_keys:
            cls._REGISTRY[k] = rule_obj

    @classmethod
    def lookup_binary(cls, op_key: str, t1, t2) -> DType | None:
        if op_key not in cls._REGISTRY:
            return None
        return cls._REGISTRY[op_key].call_binary(t1, t2)

    @classmethod
    def lookup_unary(cls, op_key: str, t) -> DType | None:
        if op_key not in cls._REGISTRY:
            return None
        return cls._REGISTRY[op_key].call_unary(t)


def select_cpp_common_int_type(t1: DType, t2: DType) -> APInt:
    assert t1.is_int_signless() and t2.is_int_signless()

    w1, w2 = t1.primitive_width, t2.primitive_width
    s1, s2 = t1.is_int(), t2.is_int()

    if s1 == s2:
        return APInt(max(w1, w2), signed=s1)

    ws = w1 if s1 else w2
    wu = w2 if s1 else w1
    if wu >= ws:
        return APInt(wu, signed=False)
    return APInt(ws, signed=True)


def select_hls_common_int_type(t1: DType, t2: DType) -> APInt:
    return select_cpp_common_int_type(t1, t2)


def _select_wider_float_type(t1: APFloat, t2: APFloat) -> APFloat:
    return t1 if t1.primitive_width >= t2.primitive_width else t2


def add_sub_rule():
    # In HLS style, index is treated as an opaque type and does not mix with
    # int/uint/float in implicit promotion rules.
    int_rules = {
        (int, int): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width) + 1
        ),
        (int, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width + 1) + 1
        ),
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width) + 1, signed=False
        ),
        (uint, int): lambda t1, t2: APInt(
            max(t1.primitive_width + 1, t2.primitive_width) + 1
        ),
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def mul_rule():
    int_rules = {
        (int, int): lambda t1, t2: APInt(t1.primitive_width + t2.primitive_width),
        (int, uint): lambda t1, t2: APInt(t1.primitive_width + t2.primitive_width),
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: APInt(
            t1.primitive_width + t2.primitive_width, signed=False
        ),
        # (uint, int): lambda t1, t2: apint(t1.primitive_width + t2.primitive_width),
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        # (IndexType, int): lambda t1, t2: apint(t1.primitive_width + t2.primitive_width),
        # (IndexType, uint): lambda t1, t2: apint(
        #     t1.primitive_width + t2.primitive_width, signed=False
        # ),
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        # covered by commutative rule
        # (apfloat, int): lambda t1, t2: t1,
        # (apfloat, uint): lambda t1, t2: t1
        # (apfloat, IndexType): lambda t1, t2: t1,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules], commutative=True
    )


def div_rule():
    int_rules = {
        (int, int): select_hls_common_int_type,
        (int, uint): select_hls_common_int_type,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_hls_common_int_type,
        (uint, int): select_hls_common_int_type,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def mod_rule():
    int_rules = {
        (int, int): select_hls_common_int_type,
        (int, uint): select_hls_common_int_type,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_hls_common_int_type,
        (uint, int): select_hls_common_int_type,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def cmp_rule():
    int_rules = {
        (int, int): select_hls_common_int_type,
        (int, uint): select_hls_common_int_type,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_hls_common_int_type,
        (uint, int): select_hls_common_int_type,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def pow_rule():
    int_rules = {
        (int, int): select_hls_common_int_type,
        (int, uint): select_hls_common_int_type,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_hls_common_int_type,
        (uint, int): select_hls_common_int_type,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {}
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def shift_rule():
    int_rules = {
        (int, int): lambda t1, t2: t1,
        (int, uint): lambda t1, t2: t1,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: t1,
        (uint, int): lambda t1, t2: t1,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    # shifting float is meaningless
    return TypingRule([int_rules, uint_rules, index_rules])


def bitwise_logic_rule():
    int_rules = {
        (int, int): select_hls_common_int_type,
        (int, uint): select_hls_common_int_type,
    }
    uint_rules = {
        (uint, uint): select_hls_common_int_type,
        (uint, int): select_hls_common_int_type,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    # bitwise/logical ops on float is meaningless
    return TypingRule([int_rules, uint_rules, index_rules], commutative=True)


def unary_invert_rule():
    int_rules = {
        (int,): lambda t: t,
    }
    uint_rules = {
        (uint,): lambda t: t,
    }
    index_rules = {
        (IndexType,): lambda t: t,
    }
    # invert on float is meaningless
    return TypingRule([int_rules, uint_rules, index_rules])


def unary_sub_rule():
    int_rules = {
        (int,): lambda t: APInt(t.primitive_width + 1),
    }
    uint_rules = {
        (uint,): lambda t: APInt(t.primitive_width + 1),
    }
    float_rules = {
        (APFloat,): lambda t: t,
    }
    return TypingRule([int_rules, uint_rules, float_rules])


def logical_op_rule():
    int_rules = {
        (int, int): lambda t1, t2: APInt(1),
        (int, uint): lambda t1, t2: APInt(1),
        (int, APFloat): lambda t1, t2: APInt(1),
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: APInt(1),
        # (uint, int): lambda t1, t2: apint(1),
        (uint, APFloat): lambda t1, t2: APInt(1),
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: APInt(1),
        # (apfloat, int): lambda t1, t2: apint(1),
        # (apfloat, uint): lambda t1, t2: apint(1)
    }
    return TypingRule([int_rules, uint_rules, float_rules], commutative=True)


def logical_not_rule():
    int_rules = {
        (int,): lambda t: APInt(1),
    }
    uint_rules = {
        (uint,): lambda t: APInt(1),
    }
    index_rules = {
        (IndexType,): lambda t: APInt(1),
    }
    float_rules = {
        (APFloat,): lambda t: APInt(1),
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def special_function_rule():
    def select_float(t):
        return fp32 if t.primitive_width <= fp32.primitive_width else fp64

    int_rules = {
        (int,): lambda t: select_float(t),
    }
    uint_rules = {
        (uint,): lambda t: select_float(t),
    }
    index_rules = {}
    float_rules = {
        (APFloat,): lambda t: t,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def max_min_rule():
    int_rules = {
        (int, int): select_hls_common_int_type,
        (int, uint): select_hls_common_int_type,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_hls_common_int_type,
        (uint, int): select_hls_common_int_type,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        )
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules], commutative=True
    )


def cpp_common_numeric_rule():
    int_rules = {
        (int, int): select_cpp_common_int_type,
        (int, uint): select_cpp_common_int_type,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_cpp_common_int_type,
        (uint, int): select_cpp_common_int_type,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): _select_wider_float_type,
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules], commutative=True
    )


def cpp_shift_rule():
    int_rules = {
        (int, int): lambda t1, t2: t1,
        (int, uint): lambda t1, t2: t1,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: t1,
        (uint, int): lambda t1, t2: t1,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    return TypingRule([int_rules, uint_rules, index_rules])


def cpp_bitwise_logic_rule():
    int_rules = {
        (int, int): select_cpp_common_int_type,
        (int, uint): select_cpp_common_int_type,
    }
    uint_rules = {
        (uint, uint): select_cpp_common_int_type,
        (uint, int): select_cpp_common_int_type,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    return TypingRule([int_rules, uint_rules, index_rules], commutative=True)


def cpp_unary_neg_rule():
    int_rules = {
        (int,): lambda t: t,
    }
    uint_rules = {
        (uint,): lambda t: t,
    }
    index_rules = {
        (IndexType,): lambda t: t,
    }
    float_rules = {
        (APFloat,): lambda t: t,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def cpp_unary_invert_rule():
    int_rules = {
        (int,): lambda t: t,
    }
    uint_rules = {
        (uint,): lambda t: t,
    }
    index_rules = {
        (IndexType,): lambda t: t,
    }
    return TypingRule([int_rules, uint_rules, index_rules])


def cpp_logical_op_rule():
    int_rules = {
        (int, int): lambda t1, t2: int1,
        (int, uint): lambda t1, t2: int1,
        (int, APFloat): lambda t1, t2: int1,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: int1,
        (uint, int): lambda t1, t2: int1,
        (uint, APFloat): lambda t1, t2: int1,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: int1,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: int1,
        (APFloat, int): lambda t1, t2: int1,
        (APFloat, uint): lambda t1, t2: int1,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules], commutative=True
    )


def cpp_logical_not_rule():
    int_rules = {
        (int,): lambda t: int1,
    }
    uint_rules = {
        (uint,): lambda t: int1,
    }
    index_rules = {
        (IndexType,): lambda t: int1,
    }
    float_rules = {
        (APFloat,): lambda t: int1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def cpp_pow_rule():
    int_rules = {
        (int, int): select_cpp_common_int_type,
        (int, uint): select_cpp_common_int_type,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_cpp_common_int_type,
        (uint, int): select_cpp_common_int_type,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): _select_wider_float_type,
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def cpp_special_function_rule():
    def select_float(t):
        return fp32 if t.primitive_width <= fp32.primitive_width else fp64

    int_rules = {
        (int,): select_float,
    }
    uint_rules = {
        (uint,): select_float,
    }
    index_rules = {
        (IndexType,): select_float,
    }
    float_rules = {
        (APFloat,): lambda t: t,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def cpp_max_min_rule():
    int_rules = {
        (int, int): select_cpp_common_int_type,
        (int, uint): select_cpp_common_int_type,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_cpp_common_int_type,
        (uint, int): select_cpp_common_int_type,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (IndexType, IndexType): lambda t1, t2: index,
    }
    float_rules = {
        (APFloat, APFloat): _select_wider_float_type,
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules], commutative=True
    )


TypeTable.register(["add", "sub"], add_sub_rule())
TypeTable.register(["mul"], mul_rule())
TypeTable.register(["div", "floordiv"], div_rule())
TypeTable.register(["mod"], mod_rule())
TypeTable.register(["pow"], pow_rule())
TypeTable.register(["eq", "ne", "lt", "le", "gt", "ge"], cmp_rule())
TypeTable.register(["lshift", "rshift"], shift_rule())
TypeTable.register(["bitwise_and", "bitwise_or", "bitwise_xor"], bitwise_logic_rule())
TypeTable.register(["neg"], unary_sub_rule())
TypeTable.register(["invert"], unary_invert_rule())
TypeTable.register(["logical_and", "logical_or"], logical_op_rule())
TypeTable.register(["logical_not"], logical_not_rule())
TypeTable.register(
    [
        "sin",
        "cos",
        "tan",
        "exp",
        "exp2",
        "log",
        "sqrt",
        "reciprocal",
        "rsqrt",
        "square",
    ],
    special_function_rule(),
)
TypeTable.register(["max", "min"], max_min_rule())

CppTypeTable.register(["add", "sub"], cpp_common_numeric_rule())
CppTypeTable.register(["mul"], cpp_common_numeric_rule())
CppTypeTable.register(["div", "floordiv"], cpp_common_numeric_rule())
CppTypeTable.register(["mod"], cpp_common_numeric_rule())
CppTypeTable.register(["pow"], cpp_pow_rule())
CppTypeTable.register(["eq", "ne", "lt", "le", "gt", "ge"], cpp_common_numeric_rule())
CppTypeTable.register(["lshift", "rshift"], cpp_shift_rule())
CppTypeTable.register(
    ["bitwise_and", "bitwise_or", "bitwise_xor"], cpp_bitwise_logic_rule()
)
CppTypeTable.register(["neg"], cpp_unary_neg_rule())
CppTypeTable.register(["invert"], cpp_unary_invert_rule())
CppTypeTable.register(["logical_and", "logical_or"], cpp_logical_op_rule())
CppTypeTable.register(["logical_not"], cpp_logical_not_rule())
CppTypeTable.register(
    [
        "sin",
        "cos",
        "tan",
        "exp",
        "exp2",
        "log",
        "sqrt",
        "reciprocal",
        "rsqrt",
        "square",
    ],
    cpp_special_function_rule(),
)
CppTypeTable.register(["max", "min"], cpp_max_min_rule())
