import ctypes
import os
import shlex
import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .base import Backend
from .utils import make_project_path

from .._C import passes
from .._C.ir import UnitAttr
from .._C.passes import emit_vivado_hls
from ..lang.core import BufferType, DType, ShapedType, TypeBase
from ..lang.kernel import Kernel

VitisMode = Literal["csim", "csyn", "sw_emu", "hw_emu", "hw"]
FlowTarget = Literal["vitis", "vivado"]

DEFAULT_DEVICE = "u280"
DEFAULT_FREQ_MHZ = 300.0
DEFAULT_VITIS_SETTINGS = Path("/opt/xilinx/2025.2/Vitis/settings64.sh")
HLS_PREPARE_PIPELINE = (
    "builtin.module(func.func(convert-linalg-to-affine-loops),canonicalize,cse)"
)
CSIM_MAKEFILE = "csim.mk"
CSIM_SHARED_LIBRARY = "libkernel.so"
TEMPLATE_DIR = Path(__file__).with_name("templates") / "vitis"

_DTYPE_TO_NP = {
    "float32": np.float32,
    "float64": np.float64,
    "index": np.int32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint1": np.bool_,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}

_DTYPE_TO_CTYPE = {
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
    "index": ctypes.c_int32,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint1": ctypes.c_bool,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
}


@dataclass(frozen=True)
class CompiledArtifacts:
    kernel_cpp: str
    kernel_h: str
    top: str


@dataclass(frozen=True)
class VitisTool:
    name: str
    executable: Path
    env: Mapping[str, str]


def _render_template(name: str, **kwargs) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8").format(**kwargs)


def _generate_run_tcl(
    top: str, part: str, freq_mhz: float, flow_target: FlowTarget
) -> str:
    clock_period = 1000.0 / freq_mhz
    return _render_template(
        "run.tcl",
        top=top,
        part=part,
        freq_mhz=freq_mhz,
        flow_target=flow_target,
        clock_period=clock_period,
    )


def _top_signature_marker(top: str) -> str:
    return f" {top}("


def _add_extern_c_to_top(hls_code: str, top: str) -> str:
    marker = _top_signature_marker(top)
    lines = []
    for line in hls_code.splitlines():
        stripped = line.lstrip()
        if (
            marker in stripped
            and (stripped.endswith(";") or stripped.endswith("{"))
            and not stripped.startswith('extern "C" ')
        ):
            indent = line[: len(line) - len(stripped)]
            line = f'{indent}extern "C" {stripped}'
        lines.append(line)
    return "\n".join(lines) + ("\n" if hls_code.endswith("\n") else "")


def _extract_top_declaration(hls_code: str, top: str) -> str:
    marker = _top_signature_marker(top)
    for line in hls_code.splitlines():
        stripped = line.strip()
        if marker in stripped and stripped.endswith(";"):
            return stripped.removeprefix('extern "C" ').strip()
    raise RuntimeError(f"Failed to find emitted declaration for top function {top}")


def _generate_kernel_header(hls_code: str, top: str) -> str:
    declaration = _extract_top_declaration(hls_code, top)
    return _render_template("kernel.h", declaration=declaration)


def _generate_csim_makefile(vitis_root: Path) -> str:
    root = os.fspath(vitis_root)
    return _render_template(
        "csim.mk",
        csim_shared_library=CSIM_SHARED_LIBRARY,
        vitis_root=root,
    )


def _prepend_env_path(env: dict[str, str], name: str, path: Path) -> None:
    old = env.get(name, "")
    env[name] = os.fspath(path) + (os.pathsep + old if old else "")


def _numpy_dtype_for_dtype(dtype: DType):
    if dtype.name not in _DTYPE_TO_NP:
        raise TypeError(f"Unsupported Vitis Python-native csim dtype: {dtype}")
    return _DTYPE_TO_NP[dtype.name]


def _ctype_for_dtype(dtype: DType):
    if dtype.name not in _DTYPE_TO_CTYPE:
        raise TypeError(f"Unsupported Vitis Python-native csim dtype: {dtype}")
    return _DTYPE_TO_CTYPE[dtype.name]


def _as_csim_array(arg, buffer_type: BufferType):
    if not isinstance(arg, np.ndarray):
        raise TypeError(
            "Vitis Python-native csim buffer arguments must be numpy arrays"
        )
    if tuple(arg.shape) != tuple(buffer_type.shape):
        raise ValueError(
            f"Expected buffer shape {tuple(buffer_type.shape)}, got {arg.shape}"
        )

    np_dtype = _numpy_dtype_for_dtype(buffer_type.dtype)
    array = arg
    if array.dtype != np_dtype:
        array = array.astype(np_dtype)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array


def _writeback_csim_arrays(arg_arrays) -> None:
    for original, array in arg_arrays:
        if isinstance(original, np.ndarray) and original is not array:
            original[...] = array.astype(original.dtype, copy=False)


def _csim_argtype(arg_type: TypeBase):
    if isinstance(arg_type, BufferType):
        return np.ctypeslib.ndpointer(
            dtype=_numpy_dtype_for_dtype(arg_type.dtype),
            ndim=len(arg_type.shape),
            flags="C_CONTIGUOUS",
        )
    if isinstance(arg_type, DType):
        return _ctype_for_dtype(arg_type)
    raise TypeError(f"Unsupported Vitis Python-native csim argument type: {arg_type}")


def _pack_csim_arg(arg, arg_type: TypeBase):
    if isinstance(arg_type, BufferType):
        array = _as_csim_array(arg, arg_type)
        return array, array
    if isinstance(arg_type, DType):
        return _ctype_for_dtype(arg_type)(arg), None
    raise TypeError(f"Unsupported Vitis Python-native csim argument type: {arg_type}")


def _csim_return_type(res_types: list[TypeBase]):
    if not res_types:
        return None
    if len(res_types) == 1 and isinstance(res_types[0], DType):
        return _ctype_for_dtype(res_types[0])
    raise TypeError("Vitis Python-native csim only supports void or scalar return")


class PythonNativeCSimulator:
    def __init__(
        self,
        *,
        top: str,
        project_path: Path,
        vitis_root: Path,
        env: Mapping[str, str],
        arg_types: list[TypeBase],
        res_types: list[TypeBase],
        make_vars: Mapping[str, str] | None = None,
    ):
        self.top = top
        self.project_path = project_path
        self.vitis_root = vitis_root
        self.env = dict(env)
        self.arg_types = list(arg_types)
        self.res_types = list(res_types)
        self.make_vars = dict(make_vars or {})
        self.library_path = project_path / CSIM_SHARED_LIBRARY
        self._library: ctypes.CDLL | None = None
        self._function = None

    def run(self, *args) -> Any:
        if len(args) != len(self.arg_types):
            raise ValueError(
                f"Expected {len(self.arg_types)} arguments, got {len(args)}"
            )
        self.build()
        func = self._get_function()
        packed_args = []
        arg_arrays = []
        for arg, arg_type in zip(args, self.arg_types):
            packed, array = _pack_csim_arg(arg, arg_type)
            packed_args.append(packed)
            if array is not None:
                arg_arrays.append((arg, array))

        result = func(*packed_args)
        _writeback_csim_arrays(arg_arrays)
        return result

    def build(self) -> Path:
        (self.project_path / CSIM_MAKEFILE).write_text(
            _generate_csim_makefile(self.vitis_root)
        )
        env = self._make_env()
        cmd = [
            "make",
            "-f",
            CSIM_MAKEFILE,
            f"TOP={self.top}",
            f"OUT={CSIM_SHARED_LIBRARY}",
            *[f"{key}={value}" for key, value in self.make_vars.items()],
        ]
        result = subprocess.run(cmd, cwd=self.project_path, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Vitis Python-native csim build failed with exit code {result.returncode}"
            )
        return self.library_path

    def _make_env(self) -> dict[str, str]:
        env = dict(self.env)
        _prepend_env_path(env, "LD_LIBRARY_PATH", self.vitis_root / "lib" / "lnx64.o")
        return env

    def _get_function(self):
        if self._function is not None:
            return self._function
        if self._library is None:
            self._library = ctypes.CDLL(os.fspath(self.library_path))
        func = getattr(self._library, self.top)
        func.argtypes = [_csim_argtype(arg_type) for arg_type in self.arg_types]
        func.restype = _csim_return_type(self.res_types)
        self._function = func
        return func


PART_NUMBERS = {
    # Embedded and Zynq.
    "ultra96v2": "xczu3eg-sbva484-1-i",
    "pynqz2": "xc7z020clg400-1",
    "zedboard": "xc7z020clg484-1",
    "zcu102": "xczu9eg-ffvb1156-2-e",
    "zcu104": "xczu7ev-ffvc1156-2-e",
    "zcu106": "xczu7ev-ffvc1156-2-e",
    "zcu111": "xczu28dr-ffvg1517-2MP-e-S",
    # Versal and Alveo.
    "vck190": "xcvc1902-vsva2197-2MP-e-S",
    "vhk158": "xcvh1582-vsva3697-2MP-e-S-es1",
    "u200": "xcu200-fsgd2104-2-e",
    "u250": "xcu250-figd2104-2L-e",
    "u280": "xcu280-fsvh2892-2L-e",
}


def _settings_from_vitis_home(vitis_home: str | None) -> Path | None:
    if not vitis_home:
        return None
    return Path(vitis_home) / "settings64.sh"


def _resolve_settings64(
    settings64: str | os.PathLike[str] | None,
    vitis_home: str | None,
) -> Path:
    if settings64:
        return Path(settings64)
    env_settings = os.getenv("VITIS_SETTINGS64")
    if env_settings:
        return Path(env_settings)
    home_settings = _settings_from_vitis_home(vitis_home or os.getenv("VITIS_HOME"))
    if home_settings:
        return home_settings
    return DEFAULT_VITIS_SETTINGS


def _source_settings_env(settings64: Path) -> dict[str, str] | None:
    if not settings64.exists():
        return None
    command = f"source {shlex.quote(str(settings64))} >/dev/null 2>&1 && env"
    result = subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


def _find_tool_in_env(env: Mapping[str, str]) -> VitisTool | None:
    path = env.get("PATH", "")
    for tool_name in ("vitis-run", "vitis_hls"):
        executable = shutil.which(tool_name, path=path)
        if executable:
            return VitisTool(tool_name, Path(executable), dict(env))
    return None


def _detect_vitis_tool(settings64: Path) -> VitisTool | None:
    sourced_env = _source_settings_env(settings64)
    if sourced_env is not None:
        tool = _find_tool_in_env(sourced_env)
        if tool is not None:
            return tool

    return _find_tool_in_env(os.environ)


class Vitis(Backend):
    name = "vitis"

    def __init__(
        self,
        kernel: Kernel,
        vitis_home: str | None = None,
        project_path: str | None = None,
        *,
        settings64: str | os.PathLike[str] | None = None,
        device: str | None = DEFAULT_DEVICE,
        part: str | None = None,
        freq_mhz: float = DEFAULT_FREQ_MHZ,
        flow: FlowTarget = "vitis",
        csim_make_vars: Mapping[str, str] | None = None,
        csim_tb_path: str | os.PathLike[str] | None = None,
    ):
        super().__init__(kernel)
        self._settings64 = _resolve_settings64(settings64, vitis_home)
        self._project_path = Path(project_path) if project_path else None
        self._vitis_home = Path(vitis_home) if vitis_home else None
        self._device = ""
        self._part = ""
        self._freq_mhz = DEFAULT_FREQ_MHZ
        self._flow: FlowTarget = "vitis"
        self._csim_make_vars = dict(csim_make_vars or {})
        self._csim_tb_path = Path(csim_tb_path) if csim_tb_path else None
        self.artifacts: CompiledArtifacts | None = None
        self.csimulator: PythonNativeCSimulator | None = None

        if part is not None:
            self.part = part
        else:
            self.device = device or DEFAULT_DEVICE
        self.freq_mhz = freq_mhz
        self.flow = flow
        self.tool = _detect_vitis_tool(self._settings64)

    @property
    def settings64(self) -> Path:
        return self._settings64

    @property
    def part(self) -> str:
        return self._part

    @part.setter
    def part(self, part: str) -> None:
        if not part:
            raise ValueError("Part number must be non-empty")
        self._part = part
        self._device = ""

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        part = PART_NUMBERS.get(device, "")
        if not part:
            raise ValueError(
                f"Unknown device {device}. Please set part number manually."
            )
        self._device = device
        self._part = part

    @property
    def freq_mhz(self) -> float:
        return self._freq_mhz

    @freq_mhz.setter
    def freq_mhz(self, freq: float) -> None:
        if freq <= 0:
            raise ValueError("Frequency must be positive")
        self._freq_mhz = float(freq)

    @property
    def flow(self) -> FlowTarget:
        return self._flow

    @flow.setter
    def flow(self, flow: FlowTarget) -> None:
        if flow not in ("vitis", "vivado"):
            raise ValueError("Flow must be either 'vitis' or 'vivado'")
        self._flow = flow

    @property
    def project_path(self) -> Path | None:
        return self._project_path

    def run(self, mode: VitisMode, *args, **kwargs) -> Any:
        if kwargs:
            raise TypeError("Vitis backend only accepts positional kernel arguments")
        if mode == "csim":
            return self.csim(*args)
        if args:
            raise TypeError("Vitis csyn does not accept runtime arguments")
        if mode == "csyn":
            return self.synth()
        raise NotImplementedError(f"Vitis mode '{mode}' is not implemented yet")

    def csim(self, *args) -> Any:
        if self._csim_tb_path is not None:
            raise NotImplementedError(
                "External tb.cpp csim mode is not implemented yet"
            )
        project_path = self.scaffold_project(overwrite=True)
        simulator = self._get_csimulator(project_path)
        return simulator.run(*args)

    def synth(self) -> Path:
        """Generate an HLS csyn project and invoke Vitis HLS."""
        project_path = self.scaffold_project()
        self._invoke_csyn(project_path)
        return project_path

    def scaffold_project(
        self, project: str | None = None, *, overwrite: bool = False
    ) -> Path:
        if project is not None:
            project_path = make_project_path(
                project, f"allo-vitis-prj-{self.kernel.func_name}", overwrite
            )
        elif self._project_path is not None:
            project_path = self._project_path
            if project_path.exists() and any(project_path.iterdir()) and not overwrite:
                raise FileExistsError(
                    f"Project path {project_path} already exists and is not empty. "
                    "Use overwrite=True to overwrite."
                )
            project_path.mkdir(parents=True, exist_ok=True)
        else:
            project_path = make_project_path(
                None, f"allo-vitis-prj-{self.kernel.func_name}", overwrite
            )

        artifacts = self._ensure_compiled()
        (project_path / "kernel.cpp").write_text(artifacts.kernel_cpp)
        (project_path / "kernel.h").write_text(artifacts.kernel_h)
        (project_path / "run.tcl").write_text(
            _generate_run_tcl(artifacts.top, self.part, self.freq_mhz, self.flow)
        )

        self._project_path = project_path
        return project_path

    def _ensure_compiled(self) -> CompiledArtifacts:
        if self.artifacts is None:
            self.artifacts = self.compile()
        return self.artifacts

    def _release_working_module(self) -> None:
        self.module = None
        self._module_owner = None

    def _get_tool(self) -> VitisTool:
        if self.tool is None:
            self.tool = _detect_vitis_tool(self._settings64)
        if self.tool is None:
            raise RuntimeError(
                "Vitis HLS tool not found. Source "
                f"{self._settings64} or pass settings64=... to the Vitis backend."
            )
        return self.tool

    def _get_vitis_env(self) -> dict[str, str]:
        sourced_env = _source_settings_env(self._settings64)
        if sourced_env is not None:
            return sourced_env
        if self.tool is not None:
            return dict(self.tool.env)
        return dict(os.environ)

    def _get_vitis_root(self) -> Path:
        if self._vitis_home is not None:
            return self._vitis_home
        if self._settings64.name == "settings64.sh":
            return self._settings64.parent
        if self.tool is not None and self.tool.executable.parent.name == "bin":
            return self.tool.executable.parent.parent
        return DEFAULT_VITIS_SETTINGS.parent

    def _get_csimulator(self, project_path: Path) -> PythonNativeCSimulator:
        if self.csimulator is not None and self.csimulator.project_path == project_path:
            return self.csimulator
        self.csimulator = PythonNativeCSimulator(
            top=self.kernel.func_name,
            project_path=project_path,
            vitis_root=self._get_vitis_root(),
            env=self._get_vitis_env(),
            arg_types=self.kernel.parse_argument_annotations(),
            res_types=self.kernel.parse_return_annotation(),
            make_vars=self._csim_make_vars,
        )
        return self.csimulator

    def _invoke_csyn(self, project_path: Path) -> None:
        tool = self._get_tool()
        run_tcl = project_path / "run.tcl"
        if tool.name == "vitis-run":
            cmd = [
                os.fspath(tool.executable),
                "--mode",
                "hls",
                "--tcl",
                "--work_dir",
                os.fspath(project_path),
                os.fspath(run_tcl),
            ]
        else:
            cmd = [os.fspath(tool.executable), "-f", "run.tcl"]

        result = subprocess.run(
            cmd,
            cwd=project_path,
            env=dict(tool.env),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Vitis HLS csyn failed with exit code {result.returncode}"
            )

    def _validate_top_abi(self) -> None:
        ret_types = self.kernel.parse_return_annotation()
        if len(ret_types) > 1:
            raise ValueError(
                "Vitis backend only supports void or a single scalar return from "
                "the top kernel. Pass output buffers as explicit arguments instead."
            )
        if ret_types and isinstance(ret_types[0], ShapedType):
            raise ValueError(
                "Vitis backend does not support returning shaped values from the "
                "top kernel. Pass the output buffer as an explicit argument instead."
            )

    def compile(self) -> CompiledArtifacts:
        if self.kernel.func_name == "kernel":
            raise ValueError(
                "'kernel' is a reserved name for Vitis HLS. Please rename your kernel function."
            )
        self._validate_top_abi()

        module = self._get_working_module()
        top_fn = module.lookup_func(self.kernel.func_name)
        if top_fn is None:
            raise RuntimeError(
                f"Kernel function {self.kernel.func_name} not found in the module"
            )
        top_fn.set_attr("top", UnitAttr.get(module.get_context()))

        passes.run(HLS_PREPARE_PIPELINE, module.get_operation())
        hls_code = emit_vivado_hls(module)
        if hls_code is None:
            raise RuntimeError("Failed to emit Vitis HLS code")
        hls_code = _add_extern_c_to_top(hls_code, self.kernel.func_name)
        top_fn = None
        module = None
        self._release_working_module()

        artifacts = CompiledArtifacts(
            kernel_cpp=hls_code,
            kernel_h=_generate_kernel_header(hls_code, self.kernel.func_name),
            top=self.kernel.func_name,
        )
        self.artifacts = artifacts
        return artifacts
