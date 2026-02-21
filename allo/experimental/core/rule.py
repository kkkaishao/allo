# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
from .types import APInt, APFloat, fp32, fp64, fp16, scalar_type, index_type, index


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

    def call_binary(self, t1, t2) -> scalar_type | None:
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

    def call_unary(self, t) -> scalar_type | None:
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
    def lookup_binary(cls, op_key: str, t1, t2) -> scalar_type | None:
        if op_key not in cls._REGISTRY:
            return None
        return cls._REGISTRY[op_key].call_binary(t1, t2)

    @classmethod
    def lookup_unary(cls, op_key: str, t) -> scalar_type | None:
        if op_key not in cls._REGISTRY:
            return None
        return cls._REGISTRY[op_key].call_unary(t)


def add_sub_rule():
    int_rules = {
        (int, int): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width) + 1
        ),
        (int, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width + 1) + 1
        ),
        (int, index_type): lambda t1, t2: APInt(
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
        (uint, index_type): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width) + 1, signed=False
        ),
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (index_type, int): lambda t1, t2: APInt(
            max(t1.primitive_width + 1, t2.primitive_width) + 1
        ),
        (index_type, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width) + 1, signed=False
        ),
        (index_type, index_type): lambda t1, t2: index,
        (index_type, APFloat): lambda t1, t2: t2,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
        (APFloat, index_type): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def mul_rule():
    int_rules = {
        (int, int): lambda t1, t2: APInt(t1.primitive_width + t2.primitive_width),
        (int, uint): lambda t1, t2: APInt(t1.primitive_width + t2.primitive_width),
        (int, APFloat): lambda t1, t2: t2,
        (int, index_type): lambda t1, t2: APInt(
            t1.primitive_width + t2.primitive_width
        ),
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: APInt(
            t1.primitive_width + t2.primitive_width, signed=False
        ),
        # (uint, int): lambda t1, t2: apint(t1.primitive_width + t2.primitive_width),
        (uint, APFloat): lambda t1, t2: t2,
        (uint, index_type): lambda t1, t2: APInt(
            t1.primitive_width + t2.primitive_width, signed=False
        ),
    }
    index_rules = {
        # (index_type, int): lambda t1, t2: apint(t1.primitive_width + t2.primitive_width),
        # (index_type, uint): lambda t1, t2: apint(
        #     t1.primitive_width + t2.primitive_width, signed=False
        # ),
        (index_type, index_type): lambda t1, t2: index,
        (index_type, APFloat): lambda t1, t2: t2,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        # covered by commutative rule
        # (apfloat, int): lambda t1, t2: t1,
        # (apfloat, uint): lambda t1, t2: t1
        # (apfloat, index_type): lambda t1, t2: t1,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules], commutative=True
    )


def div_rule():
    int_rules = {
        (int, int): lambda t1, t2: t1,
        (int, uint): lambda t1, t2: t1,
        (int, index_type): lambda t1, t2: t1,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: t1,
        (uint, int): lambda t1, t2: APInt(t1.primitive_width),
        (uint, index_type): lambda t1, t2: t1,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (index_type, int): lambda t1, t2: APInt(t1.primitive_width),
        (index_type, uint): lambda t1, t2: t1,
        (index_type, index_type): lambda t1, t2: index,
        (index_type, APFloat): lambda t1, t2: t2,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
        (APFloat, index_type): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def mod_rule():
    int_rules = {
        (int, int): lambda t1, t2: APInt(max(t1.primitive_width, t2.primitive_width)),
        (int, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width + 1)
        ),
        (int, index_type): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width + 1)
        ),
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
        (uint, int): lambda t1, t2: APInt(
            max(t1.primitive_width + 1, t2.primitive_width)
        ),
        (uint, index_type): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (index_type, int): lambda t1, t2: APInt(
            max(t1.primitive_width + 1, t2.primitive_width)
        ),
        (index_type, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
        (index_type, index_type): lambda t1, t2: index,
        (index_type, APFloat): lambda t1, t2: t2,
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
        (int, int): lambda t1, t2: APInt(max(t1.primitive_width, t2.primitive_width)),
        (int, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width + 1)
        ),
        (int, index_type): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width + 1)
        ),
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
        (uint, int): lambda t1, t2: APInt(
            max(t1.primitive_width + 1, t2.primitive_width)
        ),
        (uint, index_type): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (index_type, int): lambda t1, t2: APInt(
            max(t1.primitive_width + 1, t2.primitive_width)
        ),
        (index_type, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
        (index_type, index_type): lambda t1, t2: index,
        (index_type, APFloat): lambda t1, t2: t2,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        (APFloat, int): lambda t1, t2: t1,
        (APFloat, uint): lambda t1, t2: t1,
        (APFloat, index_type): lambda t1, t2: t1,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def pow_rule():
    def select_float(t1, _):
        return (
            fp16
            if t1.primitive_width <= fp16.primitive_width
            else (fp32 if t1.primitive_width <= fp32.primitive_width else fp64)
        )

    int_rules = {
        (int, int): select_float,
        (int, uint): select_float,
        (int, index_type): select_float,
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): select_float,
        # (uint, int): select_float,
        (uint, index_type): select_float,
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {(index_type, index_type): select_float}
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        ),
        # (apfloat, int): lambda t1, t2: t1,
        # (apfloat, uint): lambda t1, t2: t1
        # (apfloat, index_type): lambda t1, t2: t1,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules], commutative=True
    )


def shift_rule():
    int_rules = {
        (int, int): lambda t1, t2: t1,
        (int, uint): lambda t1, t2: t1,
        (int, index_type): lambda t1, t2: t1,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: t1,
        (uint, int): lambda t1, t2: t1,
        (uint, index_type): lambda t1, t2: t1,
    }
    index_rules = {
        (index_type, index_type): lambda t1, t2: index,
        (index_type, int): lambda t1, t2: index,
        (index_type, uint): lambda t1, t2: index,
    }
    # shifting float is meaningless
    return TypingRule([int_rules, uint_rules, index_rules])


def bitwise_logic_rule():
    int_rules = {
        (int, int): lambda t1, t2: APInt(max(t1.primitive_width, t2.primitive_width)),
        (int, uint): lambda t1, t2: APInt(max(t1.primitive_width, t2.primitive_width)),
        (int, index_type): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width)
        ),
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
        # (uint, int): lambda t1, t2: apint(max(t1.primitive_width, t2.primitive_width)),
        (uint, index_type): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
    }
    index_rules = {
        (index_type, index_type): lambda t1, t2: index,
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
        (index_type,): lambda t: t,
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
        (index_type,): lambda t: APInt(1),
    }
    float_rules = {
        (APFloat,): lambda t: APInt(1),
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def special_function_rule():
    def select_float(t):
        return fp32 if t.primitive_width <= fp32.primitive_width else fp64

    int_rules = {
        (int,): lambda t: select_float,
    }
    uint_rules = {
        (uint,): lambda t: select_float,
    }
    index_rules = {
        (index_type,): lambda t: select_float,
    }
    float_rules = {
        (APFloat,): lambda t: t,
    }
    return TypingRule([int_rules, uint_rules, index_rules, float_rules])


def max_min_rule():
    int_rules = {
        (int, int): lambda t1, t2: APInt(max(t1.primitive_width, t2.primitive_width)),
        (int, APFloat): lambda t1, t2: t2,
    }
    uint_rules = {
        (uint, uint): lambda t1, t2: APInt(
            max(t1.primitive_width, t2.primitive_width), signed=False
        ),
        (uint, APFloat): lambda t1, t2: t2,
    }
    index_rules = {
        (index_type, index_type): lambda t1, t2: index,
        (index_type, APFloat): lambda t1, t2: t2,
    }
    float_rules = {
        (APFloat, APFloat): lambda t1, t2: (
            t1 if t1.primitive_width >= t2.primitive_width else t2
        )
    }
    # we disable max/min for mixed int/uint types
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
