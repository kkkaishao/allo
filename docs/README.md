---
title: "Allo: An Accelerator Design Language and Compiler"
createdAt: 2026-05-09
order: 1
summary: Project overview, fork notes, documentation index, and build instructions.
keywords: ["Allo", "Hardware Accelerator", "ADL", "Compiler", "Composable Design"]
---

%toc%

# Allo: An Accelerator Design Language and Compiler

Allo is a Python-embedded, MLIR-based accelerator design language and compiler.
It is designed for modular construction of high-performance hardware
accelerators, especially machine-learning accelerators. The upstream project
combines a Python frontend, MLIR-based compiler infrastructure, scheduling and
transformation APIs, simulation flows, and backend code generation for hardware
targets such as FPGA and AI Engine platforms.

The upstream Allo repository is maintained at
[cornell-zhang/allo](https://github.com/cornell-zhang/allo). Upstream Allo is
licensed under the Apache License 2.0; this fork follows the same license.

## Upstream Project

Upstream Allo provides a unified abstraction for accelerator design and
programming. Its core features include:

- Composable behavioral and structural design, so accelerator components can be
  built independently and assembled into larger systems.
- End-to-end deployment workflows that connect Python programs, PyTorch model
  flows, simulation, verification, and hardware code generation.
- An MLIR-based compiler stack with scheduling and transformation APIs for
  specializing kernels.
- Backend support for FPGA-oriented HLS flows and AI Engine targets, with CPU
  execution primarily used for functional validation.

## This Fork

This repository is a fork of upstream Allo. The active fork branch is `allov2`,
hosted at [kkkaishao/allo](https://github.com/kkkaishao/allo). It keeps the
upstream goal of composable accelerator design, while changing the frontend and
runtime interface to make Allo kernels feel more native in Python.

The main changes in this fork are:

- A revised kernel syntax based on `@kernel`, explicit annotations, nested
  kernels, `constexpr`, `consteval`, templates, postponed type annotations, and
  clearer scoping rules.
- Better command line loggings, such as a source-aware diagnostics with
  a Clang-like style for frontend errors.
- Explicit HLS and C++ typing styles, including HLS-oriented bit growth and
  balanced-tree lowering for integer add, subtract, and multiply expressions.
- Operator infrastructure for scalar, buffer, and tensor paths, including
  arithmetic, math, and linalg operators.
- Local `Stream` frontend types for FIFO communication, including scalar and
  block payloads.
- Dataflow simulation for local streams, so producer/consumer nested kernels
  can be validated through the CPU simulation path.
- Smooth interaction between native Python code and hardware design/testing
  workflows.
- Backend and compiler infrastructure changes that support the new frontend and
  the revised simulation flow.

NOTE: This fork is still in active development, and the documentation and build
instructions may not be fully up to date.

NOTE: This fork is not intended to be merged back into the upstream repository,
and does not represent the direction of upstream development. The upstream project
is still active and have different design goals and priorities.

## Documentation Map

The documents in this directory describe the Allo frontend, compiler, and
runtime stack:

- [Frontend Syntax](frontend.md): user-facing syntax for kernels, types,
  scopes, streams, stateful variables, spatial mappings, loops, operators,
  bit manipulation, compile-time features, and diagnostics.
- [Typing Rules](typing_rules.md): type-promotion reference for the HLS and C++
  typing styles.
- [Compiler Infrastructure](compiler_infrastructure.md): developer guide for
  the frontend code generation stack, value proxies, builder APIs, the operator
  layer, and extension points.
- [Scheduling](scheduling.md): reference for the schedule API — selection,
  every transform primitive, kernel composition, streaming, backend export, and
  the scheduler internals.
- [Simulation](simulation.md): the CPU and Vitis backends, how kernels run,
  dataflow stream simulation, the calling convention, and caching.

## Installation

For the upstream released package, follow the upstream installation guide. For
this fork, building from source is the recommended path because the new frontend
and backend changes are developed in the repository.

All commands below assume a Unix-like shell. The project requires Python 3.12 or
newer, CMake 3.20 or newer, Ninja, a C++ compiler, and an LLVM/MLIR build from
the repository submodule.

## Build Instructions

### 1. Install System Dependencies

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ccache \
  clang \
  cmake \
  lld \
  ninja-build \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv
```

On macOS, install the common dependencies with Homebrew:

```bash
brew install cmake ninja python@3.12 ccache
```

On macOS, prefer the system Clang and linker or it may introduce some linking issues.

### 2. Clone This Fork

```bash
git clone https://github.com/kkkaishao/allo.git
cd allo
git checkout allov2
git submodule update --init --recursive --depth 1
```

### 3. Create a Python Environment

Using conda:

```bash
conda create -n allo python=3.12
conda activate allo
```

Using `venv`:

```bash
python3 -m venv .env --prompt allo
source .env/bin/activate
python -m pip install --upgrade pip
```

### 4. Build LLVM and MLIR

From the repository root:

```bash
cmake -S externals/llvm-project/llvm -B externals/llvm-project/build -G Ninja \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_USE_CCACHE=ON \
  -DLLVM_USE_LINKER=lld

ninja -C externals/llvm-project/build
```

On macOS, use the same source and build directories, but omit the compiler and linker-related flags:

```bash
cmake -S externals/llvm-project/llvm -B externals/llvm-project/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_USE_CCACHE=ON
  
ninja -C externals/llvm-project/build
```

### 5. Build and Install Allo

From the repository root, keep the Python environment active:

```bash
python -m pip install -v -e .
```

### 6. Build Specific Targets

After the editable install has configured the project, the CMake build directory
is `build`. Use Ninja for focused rebuilds:

```bash
ninja -C build <target>
```

Always activate the `allo` Python environment before building or running tests.
