# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .errors import ActError
from ..lang.act import (
    ActTensorType,
    ComputeSpec,
    ISA,
    IndexExpr,
    InstructionSpec,
    PatternExpr,
    TensorProxy,
)


class TextEmitter:
    def __init__(self):
        self.lines: list[str] = []
        self.indent = 0

    def emit(self, line: str = ""):
        if line:
            self.lines.append("  " * self.indent + line)
        else:
            self.lines.append("")

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def emit_isa(isa: ISA) -> str:
    emitter = TextEmitter()
    for buffer in isa.buffers:
        emitter.emit(
            f"act.buffer @{buffer.name} size({buffer.slots}) : {buffer.type_mlir()}"
        )
    if isa.buffers and isa.instructions:
        emitter.emit()

    for i, inst in enumerate(isa.instructions):
        _emit_instruction(emitter, inst)
        if i + 1 != len(isa.instructions):
            emitter.emit()
    return emitter.text()


def _emit_instruction(emitter: TextEmitter, inst: InstructionSpec):
    inst.validate_complete()
    srcs = ", ".join(f"@{buffer.name}" for buffer in inst.sources)
    dsts = ", ".join(f"@{buffer.name}" for buffer in inst.destinations)
    emitter.emit(f"act.define @{inst.name} {{")
    emitter.indent += 1
    emitter.emit(f"src({srcs}) dst({dsts})")
    _emit_addr_region(emitter, inst)
    assert inst.compute_spec is not None
    _emit_compute_region(emitter, inst.compute_spec)
    emitter.indent -= 1
    emitter.emit("}")


class AddrRegionEmitter:
    def __init__(self, emitter: TextEmitter):
        self.emitter = emitter
        self.counter = 0
        self.pattern_values: dict[int, str] = {}

    def value(self) -> str:
        name = f"%{self.counter}"
        self.counter += 1
        return name

    def emit(self, line: str = ""):
        self.emitter.emit(line)

    def materialize_index(self, expr: IndexExpr) -> str:
        if expr.kind == "param":
            return f"%{expr.value}"
        if expr.kind == "const":
            result = self.value()
            self.emitter.emit(f"{result} = arith.constant {expr.static_value} : index")
            return result
        assert expr.lhs is not None and expr.rhs is not None
        lhs = self.materialize_index(expr.lhs)
        rhs = self.materialize_index(expr.rhs)
        result = self.value()
        op = "arith.addi" if expr.kind == "add" else "arith.muli"
        self.emitter.emit(f"{result} = {op} {lhs}, {rhs} : index")
        return result

    def format_index(self, expr: IndexExpr) -> str:
        if expr.is_static:
            return str(expr.static_value)
        return self.materialize_index(expr)

    def format_paren_list(self, exprs: tuple[IndexExpr, ...]) -> str:
        return "(" + ", ".join(self.format_index(expr) for expr in exprs) + ")"

    def format_square_list(self, exprs: tuple[IndexExpr, ...]) -> str:
        return "[" + ", ".join(self.format_index(expr) for expr in exprs) + "]"

    def format_reassociation(self, reassociation: tuple[tuple[int, ...], ...]) -> str:
        groups = [
            "[" + ", ".join(str(i) for i in group) + "]" for group in reassociation
        ]
        return "[" + ", ".join(groups) + "]"

    def emit_pattern(self, pattern: PatternExpr) -> str:
        key = id(pattern)
        if key in self.pattern_values:
            return self.pattern_values[key]
        lower = pattern.pattern.lower_impl
        if lower is None:
            raise ActError(
                f"Pattern '{pattern.kind}' does not define lowering.",
                location=pattern.location,
            )
        result = lower(self, pattern)
        self.pattern_values[key] = result
        return result


def _emit_addr_region(emitter: TextEmitter, inst: InstructionSpec):
    args = ", ".join(f"%{name}: index" for name in inst.addr_params)
    emitter.emit(f"addr({args}) {{")
    emitter.indent += 1
    region = AddrRegionEmitter(emitter)
    values = [region.emit_pattern(pattern) for pattern in inst.patterns]
    types = ", ".join("!act.pattern" for _ in values)
    emitter.emit(f"act.yield {', '.join(values)} : {types}")
    emitter.indent -= 1
    emitter.emit("}")


class ComputeRegionEmitter:
    def __init__(self, emitter: TextEmitter, spec: ComputeSpec):
        self.emitter = emitter
        self.spec = spec
        self.counter = 0
        self.values: dict[TensorProxy, str] = {
            arg.proxy: f"%{arg.name}" for arg in spec.args
        }

    def value(self) -> str:
        name = f"%{self.counter}"
        self.counter += 1
        return name

    def emit(self, line: str = ""):
        self.emitter.emit(line)

    def indent(self):
        self.emitter.indent += 1

    def dedent(self):
        self.emitter.indent -= 1

    def get_value(self, proxy: TensorProxy) -> str:
        return self.values[proxy]

    def emit_region(self):
        for idx, ret in enumerate(self.spec.returns):
            dest = self.spec.args[
                len(self.spec.args) - len(self.spec.returns) + idx
            ].proxy
            self.emit_proxy(ret, final_dest=dest)
        values = [self.values[ret] for ret in self.spec.returns]
        types = ", ".join(ret.type.mlir() for ret in self.spec.returns)
        self.emitter.emit(f"act.yield {', '.join(values)} : {types}")

    def emit_proxy(
        self, proxy: TensorProxy, *, final_dest: TensorProxy | None = None
    ) -> str:
        if proxy in self.values:
            return self.values[proxy]
        node = proxy.producer
        assert node is not None
        for inp in node.inputs:
            self.emit_proxy(inp)
        lower = node.primitive.lower_impl
        if lower is None:
            raise ActError(
                f"Primitive '{node.primitive.name}' does not define lowering.",
                location=node.location or self.spec.location,
            )
        try:
            value = lower(self, node, final_dest)
        except ActError as error:
            error.attach_location(node.location or self.spec.location, override=True)
            raise
        assert node.result is not None
        self.values[node.result] = value
        return value

    def output_value(
        self, result_type: ActTensorType, final_dest: TensorProxy | None
    ) -> str:
        if final_dest is not None and final_dest.type == result_type:
            return self.values[final_dest]
        result = self.value()
        self.emitter.emit(f"{result} = tensor.empty() : {result_type.mlir()}")
        return result


def _emit_compute_region(emitter: TextEmitter, spec: ComputeSpec):
    args = ", ".join(f"%{arg.name}: {arg.type.mlir()}" for arg in spec.args)
    emitter.emit(f"compute({args}) {{")
    emitter.indent += 1
    ComputeRegionEmitter(emitter, spec).emit_region()
    emitter.indent -= 1
    emitter.emit("}")
