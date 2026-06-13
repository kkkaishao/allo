---
title: Allo Simulation
createdAt: 2026-06-13
order: 5
summary: How Allo kernels run on the CPU and Vitis backends — execution, dataflow streams, calling convention, and caching.
keywords: ["Allo", "Simulation", "CPU", "Vitis", "JIT", "CSim", "HLS"]
---

%toc%
# Simulation

An Allo kernel is a callable Python object. Calling it runs the kernel, so a
program can freely mix native Python, NumPy, ordinary helpers, and hardware-kernel
execution without changing shape. The two backends are **CPU** (a JIT path for
fast functional validation) and **Vitis** (HLS C++ codegen, C simulation,
synthesis, and hardware emulation).

There are two ways to reach a backend:

- **Direct call** runs the kernel on the CPU backend with default options.
- **`kernel.schedule().export(backend, **kwargs)`** applies any schedule, binds
  the result to the kernel, and returns a backend object you call or drive
  explicitly. This is the path used when you want a transformed kernel or the
  Vitis flow.

```python
from __future__ import annotations

import numpy as np

from allo.lang import f32, kernel

N = 64


@kernel
def vec_add(A: f32[N], B: f32[N], C: f32[N]):
    for i in range(N):
        C[i] = A[i] + B[i]


A = np.arange(N, dtype=np.float32)
B = np.arange(N, dtype=np.float32) * 10
C = np.zeros(N, dtype=np.float32)

vec_add(A, B, C)                       # CPU backend, default options
np.testing.assert_allclose(C, A + B)
```

## Running on the CPU

A direct call is shorthand for `CPU(kernel).run(*args)`. To control CPU options
or run a scheduled kernel, export an explicit backend and call it:

```python
s = vec_add.schedule()
s.pipeline(s.loop("i"), ii=1)

backend = s.export("cpu", opt_level=3)
backend(A, B, C)                       # or backend.run(A, B, C)
```

`CPU(kernel, *, opt_level=2, shared_libs=[])` lowers the kernel to LLVM through
MLIR and runs it with the MLIR `ExecutionEngine`. `opt_level` is the LLVM
optimization level; `shared_libs` adds extra runtime libraries to link. The CPU
backend is for functional validation — it executes the same numeric semantics the
hardware flow will, including local dataflow streams.

## Running on Vitis

`s.export("vitis", **kwargs)` returns a `Vitis` backend. Its constructor is:

```python
Vitis(
    kernel,
    vitis_home=None,
    project_path=None,
    *,
    device=None,
    part="xc7z020clg400-1",   # pynq-z2
    freq_mhz=300.0,
    flow="vitis",             # or "vivado"
)
```

Specify the target either by `part=` (a full part number) or `device=` (a
shorthand from the table below) — not both. `project_path` is where on-disk
project artifacts are written.

| `device`           | Part number                     |
| ------------------ | ------------------------------- |
| `ultra96v2`        | `xczu3eg-sbva484-1-i`           |
| `pynqz2`           | `xc7z020clg400-1`               |
| `zedboard`         | `xc7z020clg484-1`               |
| `zcu102`           | `xczu9eg-ffvb1156-2-e`          |
| `zcu104`, `zcu106` | `xczu7ev-ffvc1156-2-e`          |
| `zcu111`           | `xczu28dr-ffvg1517-2MP-e-S`     |
| `vck190`           | `xcvc1902-vsva2197-2MP-e-S`     |
| `vhk158`           | `xcvh1582-vsva3697-2MP-e-S-es1` |
| `u200`             | `xcu200-fsgd2104-2-e`           |
| `u250`             | `xcu250-figd2104-2L-e`          |
| `u280`             | `xcu280-fsvh2892-2L-e`          |

### HLS C++ code

`backend.hls_code` returns the generated synthesizable C++ as a string. It needs
no toolchain, so it is the fastest way to inspect codegen.

```python
code = vec_add.schedule().export("vitis").hls_code
assert "void vec_add(" in code
```

### C simulation

Calling the Vitis backend runs Python-native C simulation (csim): the generated
HLS C++ is compiled into a shared library and the top function is called directly
from Python. `backend.csim(*args)` is the explicit form; `backend(*args)` is an
alias for it.

```python
import tempfile

with tempfile.TemporaryDirectory() as proj:
    backend = vec_add.schedule().export("vitis", project_path=proj)
    backend(A, B, C)                   # csim; equivalently backend.csim(A, B, C)

np.testing.assert_allclose(C, A + B)
```

### Synthesis

`backend.synth()` scaffolds an HLS project, invokes Vitis HLS, and returns the
path to the synthesis report directory. Synthesis requires a part (pass `part=`
or `device=`).

```python
PART = "xcvu9p-flga2104-2-i"
with tempfile.TemporaryDirectory() as proj:
    report = vec_add.schedule().export("vitis", part=PART, project_path=proj).synth()
    assert report.exists()
```

### Modes and `run`

`backend.run(mode, *args)` dispatches by mode. It is the single entry point that
covers every Vitis flow:

| Mode       | Meaning                                                                   |
| ---------- | ------------------------------------------------------------------------- |
| `"csim"`   | Python-native C simulation (same as `csim(*args)` / calling the backend). |
| `"csyn"`   | HLS C-to-RTL synthesis (same as `synth()`; takes no runtime arguments).   |
| `"hw_emu"` | Hardware emulation via XRT.                                               |
| `"hw"`     | Full hardware build and on-board run.                                     |
| `"sw_emu"` | Deprecated alias; runs `csim` and warns.                                  |

`hw_emu` and `hw` build through `v++` and run an XRT host, so they need a
platform exported in the environment (`export PLATFORM=/path/to/<shell>.xpfm`).
`backend.precheck(mode)` scaffolds the kernel `.xo`, the XRT host, and (for
emulation) the emconfig, then validates the project **without** the multi-hour,
platform-locked link step — use it to confirm the frontend produced a buildable
project. csim and synthesis do not need a platform.

All run/csim/synth methods accept `exist_ok=True` (the default). Pass
`exist_ok=False` to force a rebuild of the corresponding artifacts instead of
reusing a cache entry.

### Interface configuration

Before running synthesis or hardware flows, the indexed argument interfaces can be
configured. These methods take the zero-based argument index (or `-1` for the
return value on `set_axilite`):

- `backend.set_axi(index, ...)` — AXI master (`m_axi`) for a buffer argument.
- `backend.set_axis(index, ...)` — AXI stream (`axis`) for a `Stream` argument.
- `backend.set_axilite(index, ...)` — AXI-Lite (`s_axilite`) for a scalar/buffer
  argument or the return value.

Options mirror the Vitis HLS interface pragmas. `backend.set_csim_override(**vars)`
overrides C-simulation Makefile variables (for example `cxx`, `hls_cxxflags`).

## Values and Calling Convention

Buffer arguments are passed as NumPy arrays. The backend validates the shape and
dtype against the kernel annotation and, if it must convert to a compatible
contiguous array, writes the result back to the original NumPy argument after the
run. In-place output buffers are the recommended style because they move
unchanged between CPU and Vitis.

```python
C = np.zeros(N, dtype=np.float32)
vec_add(A, B, C)                       # C is updated in place
```

Scalar arguments are passed as Python numbers and validated against the
annotation. Scalar returns are supported on both backends:

```python
@kernel
def reduce_sum(A: i32[6]) -> i32:
    s: i32 = 0
    for i in range(6):
        s = s + A[i]
    return s


total = reduce_sum(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))
```

For a Vitis top kernel, **shaped return values are rejected** — pass shaped
outputs as explicit buffer arguments instead. Non-standard `APInt` widths are
widened to the next standard width (8/16/32/64) at the host boundary, matching the
`generate-apint-wrapper` ABI.

`Stream` values are for hardware-style communication inside kernels and between
nested kernels; they are not top-level Python call arguments. Use local streams
inside the kernel and explicit NumPy buffers at the Python boundary.

## Dataflow Stream Simulation

CPU simulation supports local `Stream` values through a dataflow simulator. There
is no separate API: call the kernel normally, and the CPU lowering pipeline
rewrites stream creation, `put`, and `get` into a small host runtime.

```python
from __future__ import annotations

import numpy as np
from allo.lang import i32, kernel, Stream

@kernel
def top(x: i32[8], out: i32[8]):
    fifo: Stream[i32]

    @kernel
    def producer(src: i32[8], stream: Stream[i32]):
        for i in range(8):
            stream.put(src[i] + 1)

    @kernel
    def consumer(stream: Stream[i32], dst: i32[8]):
        for i in range(8):
            dst[i] = stream.get() * 2

    producer(x, fifo)
    consumer(fifo, out)

x = np.arange(8, dtype=np.int32)
out = np.zeros((8,), dtype=np.int32)
top(x, out)
np.testing.assert_array_equal(out, (x + 1) * 2)
```

When two or more contiguous nested-kernel calls are connected by stream
arguments, the lowering wraps them in OpenMP sections so producer and consumer
stages run concurrently. Stream lanes are bounded FIFO queues: `put` blocks when
the selected lane is full and `get` blocks when it is empty, matching hardware
FIFO behavior closely enough for functional simulation.

Scalar payloads and statically shaped block payloads are supported. A stream
array such as `Stream[i32][2, 2]` is simulated as multiple FIFO lanes selected by
the stream indices, and a shaped payload such as `Stream[i32[2, 2]]` transfers a
whole contiguous block per `put`/`get`.

Current dataflow simulation restrictions are intentionally simple:

- Stream-connected nested-kernel calls in one dataflow group must be contiguous.
- A dataflow group cannot mix stream-connected invokes with non-stream invokes.
- Stream-connected invokes in a dataflow group must not return values; pass output
  buffers explicitly.
- As with hardware FIFOs, an imbalanced producer/consumer pair can deadlock.

On the Vitis path the same local streams emit HLS streams: scalar payloads use
`hls::stream<T>` and shaped payloads use `hls::stream_of_blocks`.

## Caching

Repeated kernel runs are kept inexpensive by caching.

CPU simulation uses an in-process compile cache keyed on the kernel IR and the CPU
backend configuration (opt level, shared libs). A repeated call with the same
kernel and configuration reuses the existing MLIR execution engine.

Vitis simulation uses both an in-process cache (HLS codegen artifacts and the
loaded simulator object) and a disk-backed cache for C-simulation projects. The
CSim project is materialized under:

```text
$HOME/.allo/cache/vitis/csim/<cache-key-prefix>/
```

where the directory name is a 24-character prefix of a stable cache key computed
over the generated `kernel.cpp`, `kernel.h`, the CSim Makefile, and the detected
Vitis toolchain. The directory holds `kernel.cpp`, `kernel.h`, `csim.mk`, and a
`cache.json` recording the full key and payload. When the cached files already
exist, the build is skipped; pass `exist_ok=False` to a run/csim/synth call to
rebuild the artifacts.

## CPU Backend Internals

`Kernel.__call__` is the entry point for a direct call; it constructs a transient
`CPU(self)` and runs it. `s.export("cpu", ...)` constructs a configured `CPU`
backend over the scheduled module. In both cases the CPU backend:

1. Clones the frontend MLIR module so backend mutation does not affect the kernel.
2. Wraps non-standard `APInt` boundaries with the `generate-apint-wrapper` pass.
3. Marks the top function with the C interface attribute the execution engine
   needs.
4. Lowers local stream operations to the dataflow runtime and wraps
   stream-connected nested-kernel calls in OpenMP sections when needed.
5. Lowers the module to LLVM and builds an `ExecutionEngine` at the requested
   optimization level.
6. Packs Python scalars and NumPy arrays into the MLIR runtime ABI, invokes the
   compiled function, and writes converted arrays back to the original NumPy
   arguments.

The compiled engine is stored in the process cache, so repeated CPU runs skip
lowering and engine construction when the kernel and configuration are unchanged.

## Vitis CSim Internals

The Vitis C-simulation path is Python-native: it does not generate a host
program. Instead it turns the generated HLS C++ kernel into a shared library and
calls the top function from Python.

1. Lower the kernel to HLS C++ and generate `kernel.cpp` and `kernel.h`.
2. Materialize the CSim cache directory under `$HOME/.allo/cache`.
3. Generate a Makefile that builds a shared object with the Vitis Clang toolchain.
4. Build `libkernel.so` when it is missing or `exist_ok=False`.
5. Load the shared library with `ctypes`.
6. Configure the top function argument and return types from the kernel
   annotations.
7. Pass NumPy arrays directly to the shared library and run the function.

This gives Vitis C simulation an ordinary Python calling style — the same NumPy
buffers and scalar returns described above — while exercising the real HLS C++.
Synthesis, emulation, and hardware flows go through `v++` and a generated
Makefile/XRT host instead, driven by `synth()`, `run("hw_emu", ...)`, and
`run("hw", ...)`.
