# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-box orchestrator for the Allo LLaMA-3.2 decoder-layer FPGA accelerator.

The decoder layer is built as THREE compute units (one per U55C SLR) that hand
activations off on-chip over AXI4-Stream; only weights / IO touch HBM. This script
generates a complete, self-contained Vitis project (3 kernel .cpp + kernel.h +
system.cfg + host.cpp + Makefile + sample inputs) ready to `make`.

Usage:
  python run.py build -o PRJ_DIR [-p 1b|small] [--hf-config DIR] [--seq-len N]
                      [-q none|w4a16|w8a16] [--group-size N] [--skip-samples]

Then build/run the generated project (see the hints printed by `build`):
  cd PRJ_DIR
  export PLATFORM=/path/to/xilinx_u55c_..._xdma_3_202210_1.xpfm
  export XILINX_XRT=/opt/xilinx/xrt
  make xclbin TARGET=hw          # bitstream (deploy)
  make run    TARGET=hw_emu      # emulation (functional check, uses sample inputs)

The example is self-contained (absolute imports, no relative package), so it runs
from any working directory as `python /path/to/run.py ...`.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # cwd-independent

from config import LlamaConfig
from hardware import DEFAULT_S, MT, NT, L_LANES, generate_kernel_code
from utils import generate_harness

# A tiny config for fast hw_emu / synthesis smoke tests (the validated small_cfg).
_SMALL = dict(D=128, Dff=256, H=4, Hkv=2, dh=32, n_layers=1, vocab=256)


def load_config(args):
    if args.hf_config:
        return LlamaConfig.from_pretrained(args.hf_config)
    if args.preset == "small":
        return LlamaConfig(**_SMALL)
    return LlamaConfig()  # default: real Llama-3.2-1B dims


def print_hints(out, cfg, S):
    rel = os.path.relpath(out)
    print("\n" + "=" * 70)
    print(f"Project ready: {out}")
    print("=" * 70)
    print(f"  model : D{cfg.D} Dff{cfg.Dff} H{cfg.H} Hkv{cfg.Hkv} dh{cfg.dh}  (S={S})")
    print(f"  array : {MT}x{NT} PE tile, L={L_LANES} lanes, 3 CUs -> SLR0/1/2")
    print(
        f"  files : cu0.cpp cu1.cpp cu2.cpp kernel.h system.cfg host.cpp Makefile *.data"
    )
    print("\nBuild it (needs Vitis 2023.2 + a U55C platform):")
    print(f"  cd {rel}")
    print(f"  export PLATFORM=/path/to/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm")
    print(f"  export XILINX_XRT=/opt/xilinx/xrt")
    print(f"  make xclbin TARGET=hw         # bitstream; links at 275 MHz (FREQ_MHZ)")
    print(
        f"  make run    TARGET=hw_emu     # functional emulation with the sample inputs"
    )
    print("\nNotes:")
    print("  - system.cfg carries the HBM map, stream_connect, SLR floorplan and the")
    print("    [vivado] phys_opt / post-route-phys_opt directives for timing closure.")
    print("=" * 70)


def cmd_build(args):
    cfg = load_config(args)
    out = os.path.abspath(args.out)
    S = args.seq_len
    variant = (
        "f32" if args.quant == "none" else args.quant
    )  # w4a16 / w8a16 pass through
    gs = args.group_size if variant != "f32" else None
    qstr = variant if variant == "f32" else f"{variant} (gs{gs})"
    kw = dict(variant=variant, group_size=gs)
    print(f"[1/2] generating 3-CU kernel code (Allo, {qstr}) -> {out}")
    generate_kernel_code(cfg, out, S=S, **kw)
    print(
        f"[2/2] generating build harness (system.cfg, host.cpp, Makefile"
        f"{', sample inputs' if not args.skip_samples else ''})"
    )
    generate_harness(cfg, out, S=S, write_inputs=not args.skip_samples, **kw)
    print_hints(out, cfg, S)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate a self-contained Vitis project")
    b.add_argument("--out", "-o", required=True, help="output project directory")
    b.add_argument(
        "--preset",
        "-p",
        choices=("1b", "small"),
        default="1b",
        help="model size preset (default: 1b = real Llama-3.2-1B dims)",
    )
    b.add_argument(
        "--hf-config",
        metavar="DIR",
        help="load dims from a HF checkpoint's config.json (overrides --preset)",
    )
    b.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_S,
        help=f"prefill sequence length S (default: {DEFAULT_S}; must be a "
        f"multiple of Mt={MT})",
    )
    b.add_argument(
        "--skip-samples",
        action="store_true",
        help="skip writing *.data (a `hw` bitstream build does not need them)",
    )
    b.add_argument(
        "--quant",
        "-q",
        choices=("none", "w4a16", "w8a16"),
        default="none",
        help="weight-only quant: none=f32, w4a16=int4 / w8a16=int8 weights "
        "(activations stay f32; each weight -> (Wq, Sc, Z) triple)",
    )
    b.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="quant group size along K (must divide D and Dff; default 64)",
    )
    b.set_defaults(func=cmd_build)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
