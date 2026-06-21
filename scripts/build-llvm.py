#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
build_mlir.py — minimal, reproducible MLIR build driver for a local LLVM repo.

Usage:
    python build_mlir.py [--repo DIR] [--build DIR] [--type Release|Debug|RelWithDebInfo]
                         [--jobs N] [--ccache]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_REPO = Path(__file__).parent.parent.resolve() / "externals" / "llvm-project"
DEFAULT_BUILD = DEFAULT_REPO / "build"
DEFAULT_TYPE = "Release"

LLVM_ENABLE_PROJECTS = ["mlir"]
LLVM_TARGETS_TO_BUILD = ["Native"]

# ── helpers ───────────────────────────────────────────────────────────────────


def run(cmd: list[str], cwd: Path) -> None:
    """Run a subprocess, streaming output. Raises SystemExit on failure."""
    print(f"\n▶ {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        sys.exit(f"Error: '{tool}' not found on PATH")
    return path


# ── cmake configure ───────────────────────────────────────────────────────────


def configure(args: argparse.Namespace) -> None:
    cmake = require("cmake")
    build_dir: Path = args.build
    build_dir.mkdir(parents=True, exist_ok=True)

    # llvm/CMakeLists.txt is the entry point, not the repo root
    llvm_src = args.repo / "llvm"
    if not (llvm_src / "CMakeLists.txt").exists():
        sys.exit(f"Error: no llvm/CMakeLists.txt under {args.repo}")

    flags: dict[str, str] = {
        "CMAKE_BUILD_TYPE": args.type,
        "LLVM_ENABLE_PROJECTS": ";".join(LLVM_ENABLE_PROJECTS),
        "LLVM_TARGETS_TO_BUILD": ";".join(LLVM_TARGETS_TO_BUILD),
        "LLVM_ENABLE_ASSERTIONS": "ON",
        "LLVM_INCLUDE_EXAMPLES": "OFF",
        "LLVM_INCLUDE_TESTS": "OFF",
        "LLVM_INCLUDE_BENCHMARKS": "OFF",
        "MLIR_INCLUDE_TESTS": "OFF",
        "MLIR_ENABLE_BINDINGS_PYTHON": "ON",
        "Python3_EXECUTABLE": sys.executable,
    }

    if args.ccache and shutil.which("ccache"):
        flags["LLVM_CCACHE_BUILD"] = "ON"

    if shutil.which("clang"):
        flags["CMAKE_C_COMPILER"] = "clang"
        flags["CMAKE_CXX_COMPILER"] = "clang++"

    if shutil.which("lld"):
        flags["LLVM_USE_LINKER"] = "lld"

    cmd = [cmake, "-G", "Ninja", str(llvm_src)]

    run(cmd, cwd=build_dir)


# ── build ─────────────────────────────────────────────────────────────────────


def build(args: argparse.Namespace) -> None:
    ninja = require("ninja")
    jobs = args.jobs or os.cpu_count() or 8

    run([ninja, "-j", str(jobs)], cwd=args.build)


# ── entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Configure and build MLIR from a local LLVM repo."
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
        help="Root of the llvm-project checkout (default: script dir)",
    )
    p.add_argument(
        "--build",
        type=Path,
        default=DEFAULT_BUILD,
        help="Build directory (default: <repo>/build)",
    )
    p.add_argument(
        "--type",
        default=DEFAULT_TYPE,
        choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
    )
    p.add_argument(
        "--jobs", type=int, default=0, help="Ninja -j value (default: cpu_count)"
    )
    p.add_argument(
        "--ccache",
        action="store_true",
        help="Use ccache if available (default: off)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build(args)
    print(f"\n✅ MLIR build complete -> {args.build}")


if __name__ == "__main__":
    main()
