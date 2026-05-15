---
title: Allo Simulation
createdAt: 2026-05-09
summary: User and developer guide for the new Allo simulation flow.
keywords: ["Allo", "Simulation", "CPU", "Vitis", "JIT", "CSim"]
---

%toc%
# Simulation

This document describes the simulation interface in the new Allo frontend. The
implementation is currently staged under `allo.exp`; the examples below use the
current experimental imports and `from __future__ import annotations` for
unquoted shaped type annotations.

Simulation is designed to behave like ordinary Python execution. A kernel is a
callable Python object, and calling it runs the active simulation backend. This
lets a Python program freely mix native Python code, NumPy code, ordinary helper
functions, and hardware-kernel simulation calls without changing the shape of
the program.

## User Interface

An Allo kernel can be called directly. Without an active backend context, direct
calls use the CPU backend.

```python
from __future__ import annotations

import numpy as np

from allo.exp.lang import f32, kernel

N = 64


@kernel
def vec_add(A: f32[N], B: f32[N], C: f32[N]):
    for i in range(N):
        C[i] = A[i] + B[i]


A = np.arange(N, dtype=np.float32)
B = np.arange(N, dtype=np.float32) * 10
C = np.zeros(N, dtype=np.float32)

vec_add(A, B, C)
np.testing.assert_allclose(C, A + B)
```

The same kernel call syntax can be routed through a different simulation
backend with a context manager.

```python
from allo.exp.backend import CPU
from allo.exp.backend.vitis import Vitis


with CPU(opt_level=2):
    vec_add(A, B, C)

with Vitis():
    vec_add(A, B, C)
```

The context changes only kernel calls. Everything else remains normal Python.
This makes simulation composition natural: a Python test can preprocess inputs,
call a simulated hardware kernel, run native Python checks, call another kernel
with a different backend, and then continue with ordinary control flow.

```python
@kernel
def k1(...): ...

@kernel
def k2(...): ...

def program(...):
  A = np.zeros(...)
  B = np.ones(...)
  for i in range(10):
    for j in range(10):
      k1(...)
  if some_condition:
    k2(...)
  for i in range(10):
    with Vitis():
      # force k2 to run on Vitis here for any caller of program
      k2(...)

with CPU():
    program(...)

with Vitis(...):
    program(...)

np.testing.assert_allclose(..., ...)
```

This enables a seamless and smooth workflow for writing tests and examples that can run on both CPU and Vitis with minimal boilerplate. The same kernel call syntax works in both contexts, so users can focus on the logic of their program instead of the mechanics of switching backends.

Use an explicit backend object when the test needs backend-specific operations
or configuration.

```python
fpga = Vitis(vec_add, device="zcu102", project_path="vadd.prj")

fpga.run("csim", A, B, C)
fpga.synth()

with fpga:
    vec_add(A, B, C)
```

The context form is intended for unified simulation calls. The explicit backend
form is intended for backend-specific workflows such as Vitis HLS synthesis,
project scaffolding, or interface pragma configuration.

## Values and Calling Convention

Buffer arguments are passed as NumPy arrays. The simulation backend validates
the shape and dtype expected by the kernel annotation. If a value must be
converted to a compatible contiguous array, the backend writes the result back
to the original NumPy argument after simulation.

```python
C = np.zeros(N, dtype=np.float32)
vec_add(A, B, C)
```

In-place output buffers are the recommended style for kernels intended to move
between CPU simulation and Vitis simulation. Scalar returns are supported by the
simulation interface. For Vitis top kernels, shaped return values are currently
rejected; pass shaped outputs as explicit buffer arguments instead.

Stream values are intended for hardware-style communication inside kernels and
between nested kernels. Local `Stream` declarations lower to HLS streams in the
Vitis path, including block streams for shaped payloads. Stream-typed top-level
Python call arguments are not part of the current CPU or Python-native Vitis
CSim calling convention; use local streams inside the kernel and explicit NumPy
buffers at the Python boundary.

## Backend Contexts

Backend contexts are lightweight configuration objects. They do not permanently
bind themselves to a single kernel. When a kernel is called inside a context,
Allo creates a backend instance bound to that kernel and runs it through the
backend's `call_kernel` hook.

```python
with CPU(opt_level=2):
    vec_add(A, B, C)

# roughly:
# active_backend.call_kernel(vec_add, A, B, C)
```

Contexts can be nested. The innermost active context controls kernel calls.
Leaving the context restores the previous backend.

```python
with CPU():
    vec_add(A, B, C)

    with Vitis():
        vec_add(A, B, C)

    vec_add(A, B, C)
```

## Compilation Cache

Simulation uses caching to keep repeated kernel calls inexpensive.

CPU simulation uses an in-process compile cache. The cache key includes the
kernel IR and CPU backend configuration. A repeated call with the same kernel
and configuration reuses the existing MLIR execution engine.

Vitis simulation uses both in-process and disk-backed caches. HLS code
generation artifacts are cached in process. Python-native C simulation projects
are materialized under:

```text
$HOME/.allo/cache/vitis/csim/<cache-key-prefix>
```

The directory name is a shortened prefix of the stable cache key. The full key
and payload are stored in the cache metadata. Cache keys include the kernel IR,
generated HLS C++ source, generated kernel header, relevant backend
configuration, CSim Makefile content, and detected Vitis toolchain information.

When a cache entry is used, the log marks the stage with `(Cache Hit)`:

```text
Success Compiling CPU Kernels (Cache Hit)
Success Compiling Vitis HLS Kernels (Cache Hit)
Success Generating Vitis C Simulation Cache (Cache Hit)
Success Building Vitis C Simulation Shared Library (Cache Hit)
```

For explicit backend APIs that accept `overwrite=True`, the backend rebuilds the
corresponding generated artifacts instead of using the existing cache entry.

## CPU Simulation Internals

The CPU backend lowers the kernel to LLVM through MLIR and executes it with the
MLIR `ExecutionEngine`.

At a high level, CPU simulation does the following:

1. Clone the frontend MLIR module so backend-specific mutation does not affect
   the original kernel object.
2. Mark the top function with the C interface attribute required by the MLIR
   execution engine.
3. Lower the module to LLVM.
4. Build an `ExecutionEngine` with the requested optimization level.
5. Pack Python scalars and NumPy arrays into the ABI expected by MLIR's runtime.
6. Invoke the compiled function and write converted arrays back to the original
   NumPy arguments when needed.

The compiled engine is stored in the process cache. Repeated CPU simulation
calls therefore skip lowering and engine construction when the kernel and
configuration are unchanged.

## Vitis Python-Native CSim Internals

The Vitis context simulation path uses the Python-native C simulator. It does
not generate a host program. Instead, it turns the generated HLS C++ kernel into
a shared library and calls the top function from Python.

The flow is:

1. Lower the Allo kernel to HLS C++ and generate `kernel.cpp` and `kernel.h`.
2. Materialize a CSim cache directory under `$HOME/.allo/cache`.
3. Generate a small Makefile for building a shared object with the Vitis Clang
   toolchain.
4. Build `libkernel.so` when the shared library is missing or the user requests
   overwrite.
5. Load the shared library with `ctypes`.
6. Configure the top function argument and return types from the kernel
   annotations.
7. Pass NumPy arrays directly to the shared library and run the function.

This gives Vitis C simulation a Python-native calling style:

```python
with Vitis():
    vec_add(A, B, C)
```

Local streams in the kernel body are emitted into the generated HLS C++:
scalar payloads use `hls::stream<T>`, while shaped payloads use
`hls::stream_of_blocks`. Python arguments still follow the scalar and NumPy
buffer calling convention described above.

The current Vitis context path is intentionally limited to Python-native CSim.
HLS synthesis and project-level operations remain explicit backend operations:

```python
backend = Vitis(vec_add, device="zcu102", project_path="vadd.prj")
backend.synth()
```

## Developer Notes

`Kernel.__call__` is the single entry point for seamless simulation. It checks
the active backend context. If a context exists, it delegates to
`backend.call_kernel(kernel, *args, **kwargs)`. Otherwise, it falls back to the
CPU backend.

The backend base class owns the context mechanism. It stores the current backend
in a `ContextVar`, so nested contexts and asynchronous execution can restore the
previous active backend without relying on a process-wide mutable global.

Backends should keep context objects lightweight. A context object represents a
backend configuration, not a compiled kernel. `call_kernel` should create or
reuse a backend instance bound to the current kernel and rely on the cache layer
for performance. This avoids mixing per-kernel mutable state such as compiled
modules, execution engines, generated HLS artifacts, or loaded shared libraries
inside a long-lived context object.

The explicit backend API and the context API share the same backend
implementation. The explicit API is useful when code needs direct access to
backend-specific methods. The context API is useful when code wants ordinary
Python call syntax while switching simulation backends uniformly.
