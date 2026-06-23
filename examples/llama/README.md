# LLaMA-3.2 transformer block — prefill — on Alveo U55C (Allo)

An out-of-box demo that generates a complete, self-contained Vitis project for one
LLaMA-3.2 transformer block (a HF `LlamaDecoderLayer`) running in **prefill** mode —
processing `S` tokens at once with causal attention — as a 3-compute-unit accelerator
(one CU per U55C SLR). The CUs hand activations off **on-chip over AXI4-Stream** (no
HBM round-trip); only weights and the layer IO touch HBM.

> "decoder layer" is the *architecture* name (LLaMA is a decoder-only transformer); it
> is **not** the autoregressive *decode* phase. This build is the prefill engine; the
> single-token decode engine (KV-cache attention) is a separate module.

```
CU0 (SLR0, at HBM)  RMSNorm1 + QKV-proj + RoPE        -> Q,K,V,residual (AXIS)
CU1 (SLR1)          flash-attn + O-proj + res + RMSNorm2 -> h2, residual (AXIS)
CU2 (SLR2)          gate/up-proj + SwiGLU + down-proj + res -> out (HBM)
```

The four projections are streaming **output-stationary systolic GEMM** cores on an
8×16 (=128) PE array; attention is the dual-array flash dataflow. f32 datapath.

## Quickstart

```bash
conda activate allo

# Generate the project (Allo emits the 3 kernels; the harness is generated here).
python run.py build -o prj                 # real Llama-3.2-1B dims (D2048 ...), f32
python run.py build -o prj -p small         # tiny dims, for fast hw_emu / synth
python run.py build -o prj --hf-config /path/to/checkpoint   # any sibling model
python run.py build -o prj -q w4a16    # int4 weight-only quant (or w8a16 = int8)

# Build the generated project (Vitis 2023.2 + a U55C platform).
cd prj
export PLATFORM=/path/to/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
export XILINX_XRT=/opt/xilinx/xrt
make xclbin TARGET=hw        # bitstream; links at 275 MHz (override with FREQ_MHZ=)
make run    TARGET=hw_emu    # functional emulation, drives the generated *.data
```

## Files

| file                      | role                                                                           |
| ------------------------- | ------------------------------------------------------------------------------ |
| `config.py`               | `LlamaConfig` model dims (+ `from_pretrained`)                                 |
| `hardware.py`             | Allo kernel/schedule definitions + `generate_kernel_code(config, dir)`         |
| `utils.py`                | `generate_harness(config, dir)` → `system.cfg`, `host.cpp`, `Makefile`, inputs |
| `run.py`                  | orchestrator; `build` subcommand                                               |
| `Makefile`, `host.cpp.in` | model-independent build assets (copied / rendered into the project)            |
| `runtime.py`              | (future) XRT on-board inference for an `infer` subcommand                      |

`system.cfg` carries what Allo cannot emit: the HBM map (`sp`), the SLR floorplan,
the `stream_connect` activation handoffs, and a `[vivado]` section enabling phys_opt
+ post-route phys_opt for timing closure.

## Status / scope

- The f32 path closes near target on U55C (~290 MHz at a 300 MHz target; deployed at
  275 MHz).
- **`--quant w4a16`/`w8a16`** is weight-only quantization: each projection weight is
  stored int4/int8 and dequantized (`(Wq − Z)·Sc`, per-group scale/zero) as it streams
  into the array; activations and the accumulation stay f32 (the layer is weight-DRAM-
  bound, so the win is 4×/2× less weight traffic — fp16 activations would buy nothing
  on this fabric). Each weight becomes a `(Wq, Sc, Z)` triple → 3 m_axi ports. Numerics
  are validated by the monolith csim; the 3-CU split is the same dequant stages cut
  across SLRs. **NB int4 weights are not addressable at the XRT host boundary**, so an
  i4 `make run` is unsupported (the bitstream still builds); use `w8a16` for emulation.
- This is the **prefill** layer (S tokens). A decode engine (KV-cache attention) and
  the model-level LM head are separate, planned pieces of an end-to-end serving stack.
