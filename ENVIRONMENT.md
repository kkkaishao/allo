# Vitis/Alveo U55C Environment

**Default version:** Vitis 2022.1 (best Alveo U55C compatibility).
**Always use Docker** for Vitis commands to avoid host library conflicts:
```bash
cd /path/to/project && /path/to/docker/run-vitis.sh command
```
Docker auto-configures Vitis 2022.1, XRT, and Alveo U55C. Platform files at `/opt/xilinx/platforms`.

Direct host sourcing (non-Docker): `/tools/Xilinx/Vitis/{2022.1,2023.2,2024.2}/settings64.sh`.

# Makefile Targets (Allo-generated)

| Target        | Notes                                                |
| ------------- | ---------------------------------------------------- |
| `make csynth` | HLS only — no Docker needed, log at `hls_csynth.log` |
| `make xo`     | csynth + XO packaging - log at `hls_xo.log`          |
| `make host`   | host executable                                      |
| `make xclbin` | link XO → xclbin - log at `sys_link.log`             |
| `make all`    | xo + host + xclbin                                   |
| `make run`    | run host + xclbin  - log at `emu_run.log`            |
| `make clean`  | purge artifacts                                      |

`TARGET`: defaults to `hw`. **Prefer `hw_emu`** unless explicitly requested — `hw` bitstream takes hours.

# Typical Workflow

```python
# In Allo (Python)
mod = s.export("vitis", device="u55c")
# 1. configure axi interfaces, etc.
# 2. generate sample input with numpy for cosim (if cosim needed)
mod.scaffold_project("/path/to/prj/", *samples)
```

```bash
# Then:
cd /path/to/prj/ &&
/path/to/docker/run-vitis.sh make run TARGET=hw_emu
```
