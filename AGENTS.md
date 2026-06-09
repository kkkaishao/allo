# Building
- Always run `conda activate allo` before building or running tests
- Run `pip install -v -e .` to build the full project (includes MLIR/C++ backend)
- Run `ninja -C build [target]` to build specific targets

# Testing
- Run `python -m pytest test/` to run all tests for the new frontend
- Set `XILINX_VITIS` to any invalid path to skip tests for synthesis with Vitis to save time

# Code style
- Make small, targeted diffs rather than large refactors, and always be concise
- Prefer general solutions instead of one-off `if/else` patches
- Use Modern C++ features and best practices in C++ code

# Don'ts
- Do not modify repository structure without approval
- Do not install system packages without explicit user confirmation

# Repository structure
- Place Python frontend code in `allo/exp`
- Place MLIR dialects and passes code in `mlir/`
- Current new experimental frontend is in `allo/exp/`, while other code in `allo/` is deprecated
- Tests for new frontend lie in `test/`, while tests in `tests/` are for the deprecated frontend, and should only used as references for potential uses for the new frontend

# Notes
- Raise questions or concerns of unclear design decisions in the planning phase.
- `drafts/` folder is for temporary code when exploring new ideas, so use it for prototyping freely.
- Use `conda run -n allo` to execute commands in the `allo` environment.
