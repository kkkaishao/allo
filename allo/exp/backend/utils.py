import ctypes
import random
import string
import tempfile

from pathlib import Path

import ml_dtypes
import numpy as np


def numpy_to_ctype(np_dtype) -> type:
    """Map a numpy element dtype to the ctypes scalar used at the host boundary.

    Shared by the CPU (LLVM JIT) and Vitis C-simulation backends, which both
    marshal numpy buffers/scalars through ctypes. ``np.float16``/``bfloat16`` have
    no native ctype, so they cross as raw ``c_int16`` (the caller reinterprets the
    bits); every other dtype maps via ``np.ctypeslib.as_ctypes_type``.
    """
    np_dtype = np.dtype(np_dtype)
    if np_dtype == np.dtype(np.float16) or np_dtype == np.dtype(ml_dtypes.bfloat16):
        return ctypes.c_int16
    return np.ctypeslib.as_ctypes_type(np_dtype)


def generate_random_string(prefix: str, length: int = 8) -> str:
    """Generate a random string with the given prefix."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}-{suffix}"


def make_project_path(project: Path | str | None, prefix: str, exist_ok: bool) -> Path:
    project_path = (
        Path(project)
        if project
        else Path(tempfile.gettempdir()) / generate_random_string(prefix)
    )
    if project_path.exists() and not exist_ok and any(project_path.iterdir()):
        raise FileExistsError(
            f"Project path {project_path} already exists and is not empty."
        )
    project_path.mkdir(parents=True, exist_ok=True)
    return project_path
