---
title: Allo Frontend Syntax
createdAt: 2026-05-09
summary: Syntax reference for the new Allo Python frontend.
keywords: ["Allo", "Frontend", "Syntax", "Kernel", "DSL"]
---

%toc%
# Frontend Syntax

This document describes the new Allo frontend syntax. The implementation is
currently staged under `allo.exp`; examples that need the package namespace use
`import allo.exp as allo`.

```python
from __future__ import annotations

import allo.exp as allo
from allo.exp.lang import bool, f32, i32, u1, u32, kernel, Stream
```

The Python frontend is a restricted Python-embedded DSL (eDSL). It uses Python syntax for
readability, but only the constructs described here are part of the kernel
language.

## Kernel Definition

Allo kernels are Python functions decorated with `@kernel`. Every parameter must
have a type annotation. All examples in this document assume the Python file
starts with `from __future__ import annotations`; this lets shaped annotations
be written directly as `f32[16]` instead of quoted strings such as `"f32[16]"`.

```python
from __future__ import annotations

from allo.exp.lang import f32, kernel

@kernel
def saxpy(a: f32, x: f32[16], y: f32[16], out: f32[16]):
    for i in range(16):
        out[i] = a * x[i] + y[i]
```

Scalar annotations use type names directly. Shaped annotations use
`dtype[shape]`:

```python
@kernel
def scalar_add(x: i32, y: i32) -> i32:
    return x + y

@kernel
def vector_add(x: i32[16], y: i32[16]) -> i32[16]:
    out: i32[16] = 0
    for i in range(16):
        out[i] = x[i] + y[i]
    return out
```

Without `from __future__ import annotations`, Python evaluates annotations
before Allo sees them. In that mode, scalar annotations such as `x: i32` still
work, but shaped annotations must be quoted, for example `x: "i32[16]"`.
With postponed annotations, prefer importing Allo types into the file's scope
and using bare names such as `u32[4]`; the annotation parser resolves these
names directly.

Functions with no return value can omit the return annotation or use `-> None`.
Returning a value requires an **explicit return annotation**.

```python
@kernel
def fill(out: i32[4]):
    for i in range(4):
        out[i] = i

@kernel
def no_result(out: i32[4]) -> None:
    return
```

Multiple return values are written as tuple annotations.

```python
@kernel
def split_pair(x: i32, y: f32) -> (i32, f32):
    return x + 1, y + 1.0

@kernel
def caller(x: i32, y: f32, out: f32[1]):
    lhs, rhs = split_pair(x, y)
    out[0] = rhs + lhs
```

Return placement is **intentionally restricted**. A return may appear at the top
level of the kernel body or in a first-level `if`/`else` branch. Returns inside
loops and nested `if` statements are rejected.

```python
@kernel
def choose(cond: bool, x: i32, y: i32) -> i32:
    if cond:
        return x
    return y
```

Kernels can define nested kernels as local helper kernels. A nested kernel must
be declared at the top level of the enclosing kernel body, must use exactly one
`@kernel` decorator, and can be called like any other kernel.

```python
@kernel
def outer(x: i32, out: i32[1]):
    @kernel
    def add_one(v: i32) -> i32:
        return v + 1

    out[0] = add_one(x)
```

Nested kernel definitions are not allowed inside `if`, `for`, `grid`, or
`while` bodies. Recursive kernel calls are also rejected, including indirect
recursion across multiple top-level or nested kernels.

## Types and Annotations

The scalar types are:

| Category | Types |
| :---: | :---: |
| Signed integers | `i2` through `i16`, plus `i32`, `i64`, `i128`, `i256` |
| Unsigned integers | `u1` through `u16`, plus `u32`, `u64`, `u128`, `u256` |
| Floating point | `f16`, `f32`, `f64`, `bf16` |
| Special | `index`, `bool`, `constexpr` |

`bool` is an alias for `u1`. `index` is the preferred type for loop indices and
values used as dynamic indices.

Use `apint(width, signed=False)` to define custom integer widths beyond the
predefined aliases. Unsigned custom integers are the default; pass `signed=True`
for signed integers.

```python
from allo.exp.lang import apint, kernel

u17 = apint(17)
i23 = apint(23, signed=True)

@kernel
def custom_width(x: u17, y: i23, out: u17[1]):
    out[0] = x + y
```

Shaped values are written as `dtype[shape]`. With postponed annotations, a
rank-0 shaped value is written as `dtype[()]` because Python does not allow an
empty subscript expression. The quoted spelling `"dtype[]"` is still accepted.

```python
@kernel
def shapes(a: f32[8], b: i32[4, 4], acc: f32[()]):
    acc[()] = a[0] + b[0, 0]
```

Shape expressions are compile-time integer expressions. They may use integer
literals, visible constants, template parameters, unary `+`/`-`, and the binary
operators `+`, `-`, `*`, and `//`.

```python
M = 4
N = 8

@kernel
def reshape_like(inp: i32[M * N], out: i32[M, N]):
    for i, j in allo.grid(M, N):
        out[i, j] = inp[i * N + j]
```

By default, shaped annotations describe mutable buffers. With
`KernelOptions(enable_tensor=True)`, the same annotation syntax describes MLIR
tensors.

```python
from allo.exp.lang import KernelOptions

@kernel(options=KernelOptions(enable_tensor=True))
def tensor_add(x: f32[4], y: f32[4]) -> f32[4]:
    return x + y
```

### Streams

`Stream` describes local FIFO channels. The payload type can be a scalar dtype
or a shaped buffer payload. The optional second bracket group describes an array
of streams; omitting it creates a single rank-0 stream. The current frontend
uses a default stream depth of `2`.

```python
from allo.exp.lang import Stream, i32

@kernel
def scalar_stream(x: i32, out: i32[1]):
    fifo: Stream[i32]
    fifo.put(x)
    out[0] = fifo.get()

@kernel
def stream_array(x: i32, out: i32[1]):
    fifo: Stream[i32][2, 2]
    fifo[0, 1].put(x)
    out[0] = fifo[0, 1].get()
```

A stream with a shaped payload transfers a whole block. In Vitis HLS emission,
scalar payloads map to `hls::stream<T>` and shaped payloads map to
`hls::stream_of_blocks<T[...], depth>`.

```python
@kernel
def block_stream(out: i32[1]):
    fifo: Stream[i32[4, 4]]
    buf: i32[4, 4]
    buf[0, 0] = 7
    fifo.put(buf)
    recv = fifo.get()
    out[0] = recv[0, 0]
```

Streams can be passed explicitly to nested kernels. This is the supported way to
connect producer and consumer stages inside one top-level kernel.

```python
@kernel
def nested_stream(x: i32, out: i32[1]):
    fifo: Stream[i32]

    @kernel
    def producer(v: i32, stream: Stream[i32]):
        stream.put(v + 1)

    @kernel
    def consumer(stream: Stream[i32], dst: i32[1]):
        dst[0] = stream.get()

    producer(x, fifo)
    consumer(fifo, out)
```

Streams must be declared without initializers. A stream array must be indexed
with exactly one scalar index per stream dimension before `get()` or `put()`.
Stream references are not assignable; use `put(value)` to write and `get()` to
read. Stream values are not valid kernel return values. `GStream` global stream
syntax is not part of the current public frontend surface.

## Variables and Scope

Annotated assignments declare variables.

```python
@kernel
def declarations(x: i32, out: i32[4]):
    base: i32 = x
    tmp: i32[4] = 0
    for i in range(4):
        tmp[i] = base + i
        out[i] = tmp[i]
```

Shaped locals may be declared without an initializer. This allocates a local
buffer in the default mode or an empty tensor in tensor mode.

```python
@kernel
def local_buffer(out: i32[4]):
    buf: i32[4]
    for i in range(4):
        buf[i] = i
        out[i] = buf[i]
```

Scalar variables must be initialized when declared. A runtime local can also be
introduced by assigning an existing runtime value.

```python
@kernel
def inferred_local(cond: bool, x: i32, y: i32, out: i32[1]):
    v = x
    if cond:
        v = y
    else:
        v = x + y
    out[0] = v
```

Compile-time variables must be declared with `constexpr`. They are evaluated
during compilation and cannot be reassigned.

```python
from allo.exp.lang import constexpr

@kernel
def constexpr_bound(out: i32[4]):
    N: constexpr = 4
    for i in range(N):
        out[i] = i
```

List initializers are supported for shaped values when every element is a
compile-time `int` or `float`. The list shape must exactly match the annotation.

```python
@kernel
def constants(out: i32[2, 2]):
    scale: constexpr = 3
    table: i32[2, 2] = [[1, scale], [scale + 1, scale + 2]]
    for i, j in allo.grid(2, 2):
        out[i, j] = table[i, j]
```

Allo uses block scope. Variables declared inside an `if`, `for`, `grid`, or
`while` body are local to that block. Declare a variable before the block if it
must be used afterward.

```python
@kernel
def scoped(cond: bool, x: i32, out: i32[1]):
    value: i32 = 0
    if cond:
        value = x
    else:
        value = x + 1
    out[0] = value
```

A name cannot be redeclared in the same scope. Later assignments are cast back to
the variable's original type.

Nested kernels follow the same scoping model, but their captures are deliberately
limited. They may capture compile-time symbols from the enclosing scope:
`constexpr` values, concrete types, type aliases, other kernels, `consteval`
functions, Allo operators, and modules. They may not capture runtime values such
as enclosing kernel parameters, local scalar variables, loop indices, or buffers.
Pass runtime values explicitly as nested-kernel arguments.

```python
@kernel
def captures(x: i32, out: i32[1]):
    offset: constexpr = 2
    T: constexpr = i32

    @kernel
    def add_offset(v: T) -> T:
        return v + offset

    out[0] = add_offset(x)
```

## Loops

Both Python `range` and `allo.range` are supported in kernels. They accept the
same one-, two-, or three-argument forms.

```python
from allo.exp.lang import range as allo_range


@kernel
def ranges(out: i32[20]):
    for i in range(10):
        out[i] = i
    for i in range(10, 20):
        out[i] = i
    for i in allo_range(0, 20, 2):
        out[i] = i * 2
```

Loop bounds may depend on runtime values. Loop steps must be positive if they are not `constexpr`.

```python
@kernel
def variable_bounds(a: i32[10], out: i32[10]):
    for i in range(10):
        for j in range(a[i], 10, a[i]):
            out[j] += i
```

`allo.grid` is a shorthand for a multidimensional parallel loop. It requires at
least two dimensions, and the loop target must be a tuple with the same number of
variables.

```python
@kernel
def matmul(a: f32[32, 32], b: f32[32, 32]) -> f32[32, 32]:
    c: f32[32, 32] = 0.0
    for i, j in allo.grid(32, 32):
        for k in range(32):
            c[i, j] += a[i, k] * b[k, j]
    return c
```

Grid dimensions may also be written as `(start, stop)` or
`(start, stop, step)` tuples.

```python
@kernel
def strided_grid(out: i32[8, 8]):
    for i, j in allo.grid((0, 8, 2), (1, 8, 2)):
        out[i, j] = i + j
```

At the moment, `grid` does not support non-trivial loop-carried scalar
dependencies. Use nested `range` loops when the loop body needs to update a
scalar accumulator across iterations.

`while` loops are supported for runtime conditions. A `while` loop may update
loop-carried scalar values.

```python
@kernel
def count(out: i32[1]):
    i: i32 = 0
    acc: i32 = 0
    while i < 4:
        acc += i
        i += 1
    out[0] = acc
```

`break`, `continue`, `for ... else`, and `while ... else` are not supported.

## Conditionals

Runtime `if`/`elif`/`else` statements lower to structured control flow. Variables declared
outside the conditional can be assigned in either branch and used afterward.

```python
@kernel
def classify(x: i32, y: i32) -> i32:
    result: i32 = 0
    if x == 0:
        result = 1
    elif y > x:
        result = 2
    else:
        result = 3
    return result
```

Conditions may use comparison operators, `and`, `or`, and `not`.

```python
@kernel
def logic(a: i32[3], b: i32) -> i32:
    out: i32 = 0
    if a[0] > 0 and b < 0:
        out = 1
    elif a[1] <= 1 or not (a[2] == 3):
        out = 2
    return out
```

Ternary expressions lower to a select operation when the condition is runtime.
At least one branch must be a runtime value so the result type can be inferred.

```python
@kernel
def select(cond: bool, x: i32, y: i32) -> i32:
    return x if cond else y
```

If a condition is a `constexpr`, the frontend evaluates the condition during
compilation and only emits the selected branch.

## Expressions and Operators

The frontend supports the following Python operators.

| Category | Operators |
| :---: | :---: |
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%`, `**` |
| Unary | `+x`, `-x`, `~x`, `not x` |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Boolean | `and`, `or` |
| Bitwise | `&`, `\|`, `^`, `<<`, `>>` |
| Assignment | `=`, `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `&=`, `\|=`, `^=`, `<<=`, `>>=` |

Multi-way comparisons such as `a < b < c` are not supported; write them with
`and`.

```python
@kernel
def comparisons(a: i32, b: i32, c: i32) -> bool:
    return a < b and b < c
```

The default `typing_style` is `"hls"`, which uses HLS-oriented integer
promotion. For example, an addition may widen internally and then cast back to
the destination type. `KernelOptions(typing_style="cpp")` selects C++-style
promotion rules.

```python
@kernel(options=KernelOptions(typing_style="cpp"))
def cpp_style(x: u32, y: i32, out: u32[1]):
    out[0] = x + y
```

`min` and `max` are supported as built-ins and lower to Allo arithmetic
operators.

```python
@kernel
def clamp(x: i32, lo: i32, hi: i32) -> i32:
    return min(max(x, lo), hi)
```

Only Allo kernels, Allo operators, and `consteval` functions may be called from
inside a kernel. The static built-ins `print` and `len` are evaluated during
compilation when their arguments are compile-time values.

## Indexing and Memory Access

Shaped values use tuple-style indexing. The number of indices must match the
rank.

```python
@kernel
def copy_2d(src: f32[4, 4], dst: f32[4, 4]):
    for i, j in allo.grid(4, 4):
        dst[i, j] = src[i, j]
```

Rank-0 shaped values are indexed with `()`.

```python
from allo.exp.lang import KernelOptions
from allo.exp.operators import linalg


@kernel(options=KernelOptions(enable_tensor=True))
def dot_scalar(a: f32[4], b: f32[4]) -> f32:
    return linalg.dot(a, b)[()]
```

Integer scalar values support single-bit extraction and insertion with
subscript syntax.

```python
@kernel
def bit_ops(x: u32, out: u1[1]):
    out[0] = x[0]
```

Python slice indices such as `A[0:4]`, partial subviews such as `A[i]` for a
rank-2 buffer, and bit ranges such as `x[0:4]` are not part of the current new
frontend.

## Operator Namespaces

Python operators cover scalar arithmetic and shaped elementwise expressions.
Explicit operator calls are useful when an operation needs an output
accumulator. The current experimental operator modules live under
`allo.exp.operators`.

```python
from allo.exp.operators import arith, linalg, math


@kernel
def memref_elementwise(x: f32[4], y: f32[4], out: f32[4]):
    arith.add(x, y, acc=out)
```

Math operators include `exp`, `exp2`, `log`, `log2`, `abs`, `pow`, `sqrt`,
`rsqrt`, `sin`, `cos`, `tan`, `floor`, `ceil`, and `erf`. They work on scalar
values and on shaped values.

```python
@kernel
def sigmoid(x: f32[8], out: f32[8]):
    for i in range(8):
        out[i] = 1.0 / (1.0 + math.exp(-x[i]))
```

Linalg operators currently include `matmul` and `dot`. They support both
default buffer mode and tensor mode. In buffer mode, pass an explicit `acc=`
output because the operation writes into an existing buffer. In tensor mode, the
same operation can return a tensor value directly.

```python
@kernel(options=KernelOptions(enable_tensor=True))
def dense(a: f32[2, 3], b: f32[3, 4]) -> f32[2, 4]:
    return linalg.matmul(a, b)
```

```python
@kernel
def buffer_matmul(a: f32[2, 3], b: f32[3, 4], out: f32[2, 4]):
    linalg.matmul(a, b, acc=out)
```

## Compile-Time Features

Global Python `int` and `float` values are visible as compile-time constants.

```python
SCALE = 3


@kernel
def add_scale(x: i32) -> i32:
    return x + SCALE
```

`consteval` marks a Python helper function that runs during compilation.

```python
from allo.exp.lang import consteval


@consteval
def factor():
    return 3


@kernel
def use_factor(x: i32) -> i32:
    return x + factor()
```

Templates parameterize kernels over compile-time types and values. A templated
kernel is not concrete until it is specialized with `kernel[...]`.

```python
from allo.exp.lang import Template, f32, i32

T = Template("T")
N = Template("N")

@kernel(T, N)
def fill_template(x: T, out: T[N]):
    for i in range(N):
        out[i] = x

fill_i32_4 = fill_template[i32, 4]
```

Template bindings must be provided before compilation or execution. Type
templates can be used in scalar annotations and as the head of shaped
annotations. Integer templates can be used in shape expressions and loop bounds.

Templates are different from ordinary global aliases. A global alias such as
`T = i32` is an immediately chosen concrete type; every use of `T` in that kernel
means `i32`, and callers cannot specialize it. A `Template("T")` is a delayed
binding point that must be supplied by the caller.

```python
FixedT = i32

@kernel
def fixed_alias(x: FixedT, out: FixedT[4]):
    for i in range(4):
        out[i] = x

T = Template("T")

@kernel(T)
def delayed_type(x: T, out: T[4]):
    for i in range(4):
        out[i] = x


delayed_i32 = delayed_type[i32]
delayed_f32 = delayed_type[f32]
```

## Diagnostics

The new frontend reports compilation errors in a clang-like style. Diagnostics
include the source file, line, column, error message, the relevant source line,
and a caret span pointing at the AST node that triggered the error.

For example, an undefined name in:

```python
def broken(x):
    return x + y
```

is rendered as:

```text
broken.py:11:16: error: Name 'y' is not defined
11 |     return x + y
   |                ^
```

The same format is used for kernel syntax errors such as missing annotations,
unsupported control flow, return type mismatches, illegal captures, or invalid
operator calls. When an error occurs while compiling a called kernel or nested
kernel, the diagnostic message is wrapped with call context so the caller and
callee relationship is visible.

Source locations are based on Python source inspection. They are reliable for
kernels defined in normal `.py` files. In a REPL, notebook, `python -c`, or other
dynamically generated context, Python may not expose stable source lines; Allo
will still report the error, but file names and line numbers can be missing or
inaccurate. For compiler debugging, set `ALLO_SHOW_COMPILER_TRACEBACK=1` to keep
the full Python traceback instead of the shortened user diagnostic.

## Common Restrictions

The new frontend intentionally rejects unsupported Python early and reports the
source location. The most important restrictions are:

- All kernel parameters require explicit annotations.
- Returning a value requires an explicit return annotation.
- `return` is not supported inside loops or nested `if` statements.
- `break`, `continue`, loop `else` blocks, arbitrary Python calls, attribute
  assignment, chained assignment such as `a = b = c`, and multi-way comparisons
  are not supported.
- `constexpr` variables must be explicitly annotated, initialized at
  declaration, and never reassigned.
- Runtime values from an outer kernel scope cannot be captured by nested
  kernels.
- `Stream` may be declared in a kernel body and passed explicitly to nested
  kernels, but it cannot be returned from a kernel.
- Recursive kernel calls, including indirect recursion through nested kernels,
  are not supported.
- Python slices, partial tensor subviews, dynamic `...` shapes, tensor methods
  such as `.T` and `.copy()`, and bit-range indexing are not supported in the
  current new frontend.

## Differences from the Upstream Frontend

The upstream frontend in `docs/source/dive/frontend_syntax.rst`, `tests/`, and
`examples/` describes the older API. It is useful as historical context, but the
new frontend should be documented from `test/`, `allo/exp`, and `example/`.

| Area | Older upstream frontend | New frontend |
| --- | --- | --- |
| Kernel entry | Plain Python function passed to `allo.customize` | Function decorated with `@kernel` |
| Import style | `from allo.ir.types import int32, float32, ConstExpr` | `from __future__ import annotations`; `from allo.exp.lang import i32, f32, constexpr, kernel` |
| Shaped annotations | Often `int32[32, 32]` | Postponed annotations such as `i32[32, 32]`; quoted strings are still accepted |
| Compile-time constants | `ConstExpr[...]` | `constexpr` annotation and `@consteval` helpers |
| Templates | Old Python generic syntax and scheduler instantiation | `Template("T")`, `@kernel(T)`, and `kernel[i32]` specialization |
| Kernel calls | Helper functions inside customized functions | Calls between `@kernel` functions, including nested kernels |
| Diagnostics | Many errors surfaced later or through Python exits | Frontend diagnostics point to source locations |
| Default shaped value | Old tensor/memref behavior from upstream schedule flow | Mutable buffer by default; tensor mode with `KernelOptions(enable_tensor=True)` |
| Streams | Older dataflow APIs and scheduling-driven stream insertion | Local `Stream` declarations with scalar/block payloads and explicit nested-kernel stream parameters |

Some old examples are not current syntax:

```python
# Old style
from allo.ir.types import int32

def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    ...

s = allo.customize(gemm)
```

The new form is:

```python
from __future__ import annotations

from allo.exp.lang import f32, kernel

@kernel
def gemm(A: f32[32, 32], B: f32[32, 32]) -> f32[32, 32]:
    C: f32[32, 32] = 0.0
    for i, j in allo.grid(32, 32):
        for k in range(32):
            C[i, j] += A[i, k] * B[k, j]
    return C
```

Features documented in the old frontend guide should not be copied into new
documentation unless they are implemented in the new frontend. In particular,
old `meta_if`/`meta_for`, dynamic `float32[...]` shapes, partial subviews,
general Python slicing, tensor attributes such as `.T`, `.copy`, and `.reverse`,
old fixed-point type attributes, global `GStream` syntax, and high-level
neural-network library calls are outside the currently documented new frontend
surface.
