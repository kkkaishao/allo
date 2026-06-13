# Building
- Always run `conda activate allo` before building or running tests
- Run `pip install -v -e .` to build the full project (includes MLIR/C++ backend)
- Run `ninja -C build [target]` to build specific targets

# Testing
- Run `python -m pytest tests/` to run all tests
- Set `XILINX_VITIS` to any invalid path to skip tests for synthesis with Vitis to save time

# Running
- Use `conda run -n allo` to execute commands in the `allo` environment.
- When the host system is not compatible with a specific Vitis version,
  use `docker/run-vitis.sh <command>` to run commands in a docker container.

# Code style
- Make small, targeted diffs rather than large refactors, and always be concise
- Prefer general solutions instead of one-off `if/else` patches
- Use Modern C++ features and best practices in C++ code
- Use `assert` to enforce invariants and assumptions that should always hold by the design,
  and fail loudly during development instead of being silently tolerated.

# Don'ts
- Do not modify repository structure without approval
- Do not install system packages without explicit user confirmation

# Repository structure
- Place Python frontend code in `allo/`
- Place MLIR dialects and passes code in `mlir/`
- Tests lie in `tests/`
- Use `drafts/` for temporary code when exploring new ideas
