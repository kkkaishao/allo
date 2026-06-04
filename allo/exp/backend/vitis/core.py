from __future__ import annotations

import functools
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Generic, TypeVar, ParamSpec

from ..base import (
    Backend,
    lookup_kernel,
    run_pipeline,
    stable_cache_hash,
    text_hash,
    write_json_if_changed,
    write_text_if_changed,
)
from ..utils import make_project_path
from .csim import (
    CSIM_MAKEFILE,
    CsimToolchain,
    PythonNativeCSimulator,
    _generate_csim_makefile,
    discover_csim,
)
from .report import VitisSynthReport
from .utils import (
    VitisTool,
    _INTERFACE_MODES,
    _add_extern_c_to_top,
    _apply_interface_pragmas,
    detect_vitis_tool,
    generate_kernel_header,
    generate_run_tcl,
    _is_stream_type,
    _log_synth_failure,
    _log_synth_note,
    _normalize_interface_options,
    _probe_vitis_tool,
    _source_settings_env,
    _synth_log_path,
)
from ...._mlir import ir
from ...._mlir.dialects.allo import emit_vivado_hls
from ...._mlir._mlir_libs._allo import ir_ext
from ...lang.core import BufferType, ShapedType, TypeBase
from ...lang.kernel import Kernel
from ...logging import run_command, stage, terminate_on_error

VitisMode = Literal["csim", "csyn", "sw_emu", "hw_emu", "hw"]
FlowTarget = Literal["vitis", "vivado"]
AxiOffset = Literal["off", "direct", "slave"]
AxisRegisterMode = Literal["forward", "reverse", "both", "off"]
AxiliteStorageImpl = Literal["auto", "bram", "uram"]

DEFAULT_DEVICE = "u280"
DEFAULT_FREQ_MHZ = 300.0
DEFAULT_VITIS_HOME = Path("/opt/xilinx/2025.2/Vitis")
HLS_PREPARE_PIPELINE = """
builtin.module(
grid-mapping,
materialize-topology,
canonicalize,
cse,
convert-allo-to-func,
func.func(convert-linalg-to-affine-loops),canonicalize,cse)
"""
CSIM_CACHE_DIR_KEY_LENGTH = 24
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
# default to pynq-z2
DEFAULT_PART = "xc7z020clg400-1"


@dataclass(frozen=True)
class CompiledArtifacts:
    kernel_cpp: str
    kernel_h: str
    top: str


@dataclass(frozen=True)
class InterfacePragma:
    mode: str
    options: Mapping[str, str | int | bool | None]


def _collect_interface_options(
    named: Mapping[str, str | int | bool | None],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge named interface options (dropping ``None``) with raw keyword options."""
    options: dict[str, Any] = {
        name: value for name, value in named.items() if value is not None
    }
    options.update(extra)
    return options


def _normalize_csim_make_vars(
    overrides: Mapping[str, object | None],
) -> dict[str, str | None]:
    """Normalize ``set_csim_override`` keywords to Makefile assignments: the key
    uppercases to the variable name (``cxx`` -> ``CXX``, ``hls_cxxflags`` ->
    ``HLS_CXXFLAGS``), PathLike values become strings, and ``None`` is preserved
    (it drops a previously set override)."""
    normalized: dict[str, str | None] = {}
    for key, value in overrides.items():
        if not key:
            raise ValueError("CSim override keys must be non-empty")
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        normalized[key.upper()] = None if value is None else str(value)
    return normalized


P = ParamSpec("P")
R = TypeVar("R")


def _detect_vitis_home(vitis_home: str | None) -> Path:
    if vitis_home:
        return Path(vitis_home)
    vitis_env = os.environ.get("XILINX_HLS") or os.environ.get("XILINX_VITIS")
    if vitis_env:
        return Path(vitis_env)
    return DEFAULT_VITIS_HOME


@functools.cache
def is_vitis_available(vitis_home: str | None = None) -> bool:
    """Whether a Vitis HLS toolchain can be detected, as a plain cached bool.

    Unlike ``detect_vitis_tool`` this never raises and emits no logs, so it is
    safe to use directly in ``pytest.mark.skipif`` predicates."""
    settings64 = _detect_vitis_home(vitis_home) / "settings64.sh"
    try:
        _probe_vitis_tool(settings64)
        return True
    except Exception:
        return False


class Vitis(Backend, Generic[P, R]):
    name = "vitis"
    part: str
    tool: VitisTool

    @terminate_on_error
    def __init__(
        self,
        kernel: Kernel[P, R],
        vitis_home: str | None = None,
        project_path: str | None = None,
        *,
        device: str | None = None,
        part: str | None = DEFAULT_PART,
        freq_mhz: float = 300.0,
        flow: FlowTarget = "vitis",
    ):
        super().__init__(kernel)
        # setup toolchain paths
        self._settings64 = _detect_vitis_home(vitis_home) / "settings64.sh"
        self._vitis_home = Path(vitis_home) if vitis_home else None
        # setup project related settings
        self._project_path = Path(project_path) if project_path else None
        self.freq_mhz = freq_mhz
        self.flow: FlowTarget = flow
        self._csim_make_vars: dict[str, str] = {}
        self._init_part_and_device(part, device)
        # interface pragmas set by the user
        self._interface_pragmas: dict[int, dict[str, InterfacePragma]] = {}
        # intermediate stuffs
        self.artifacts: CompiledArtifacts | None = None
        self.csimulator: PythonNativeCSimulator | None = None
        self._csim_toolchain: CsimToolchain | None = None

    def _init_part_and_device(self, part: str | None, device: str | None) -> None:
        if part is not None and device is not None:
            raise ValueError("Cannot specify both part number and device")
        if device is not None:
            part = PART_NUMBERS.get(device)
            if not part:
                raise ValueError(
                    f"Unknown device {device}. Please specify part number."
                )
            self.part = part
            return
        if part is not None:
            self.part = part
            return
        self.part = DEFAULT_PART

    def _require_vitis_tool(self):
        if not hasattr(self, "tool") or self.tool is None:
            self.tool = detect_vitis_tool(self._settings64)

    @property
    def hls_code(self) -> str:
        artifacts = self._ensure_compiled()
        return artifacts.kernel_cpp

    @terminate_on_error
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.csim(*args, **kwargs)

    def set_csim_override(self, **overrides: str | None):
        """Override C simulation Makefile variables (e.g. ``cxx``, ``hls_cxxflags``).

        Keys are matched case-insensitively against the Makefile variables in
        ``csim.mk`` (``cxx`` -> ``CXX``, ``hls_cxxflags`` -> ``HLS_CXXFLAGS``,
        ...). Pass ``None`` to drop a previously set override.
        """
        for key, value in _normalize_csim_make_vars(overrides).items():
            if value is None:
                self._csim_make_vars.pop(key, None)
            else:
                self._csim_make_vars[key] = value
        self.csimulator = None

    #################################
    # Interface configuration methods
    #################################

    def set_axi(
        self,
        index: int,
        *,
        bundle: str | None = None,
        depth: int | None = None,
        offset: AxiOffset | None = None,
        channel: str | None = None,
        latency: int | None = None,
        num_read_outstanding: int | None = None,
        num_write_outstanding: int | None = None,
        max_read_burst_length: int | None = None,
        max_write_burst_length: int | None = None,
        max_widen_bitwidth: int | None = None,
        alignment_byte_size: int | None = None,
        name: str | None = None,
        **kwargs: str | int | bool | None,
    ):
        """
        Set the indexed argument to be an AXI master interface with the given options. For details of the options, please refer to [Vitis HLS Interface Pragma](https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-interface)

        In Vitis HLS, AXI master interfaces can only be applied to pointer (buffer) arguments.
        """
        arg_type = self._validate_interface_index(index, allow_return=False)
        if not isinstance(arg_type, BufferType):
            raise ValueError(
                "Vitis m_axi interface can only be set on buffer arguments"
            )

        options = _collect_interface_options(
            {
                "bundle": bundle,
                "depth": depth,
                "offset": offset,
                "channel": channel,
                "latency": latency,
                "num_read_outstanding": num_read_outstanding,
                "num_write_outstanding": num_write_outstanding,
                "max_read_burst_length": max_read_burst_length,
                "max_write_burst_length": max_write_burst_length,
                "max_widen_bitwidth": max_widen_bitwidth,
                "alignment_byte_size": alignment_byte_size,
                "name": name,
            },
            kwargs,
        )
        self._set_interface_pragma(index, "m_axi", options)

    def set_axis(
        self,
        index: int,
        *,
        register: bool | None = None,
        register_mode: AxisRegisterMode | None = None,
        depth: int | None = None,
        name: str | None = None,
        bundle: str | None = None,
        **kwargs: str | int | bool | None,
    ):
        """
        Set the indexed stream argument to be an AXI stream interface with the given options. For details of the options, please refer to [Vitis HLS Interface Pragma](https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-interface)

        In Vitis HLS, AXI stream interfaces can only be applied to ``Stream`` arguments.
        """
        arg_type = self._validate_interface_index(index, allow_return=False)
        if not _is_stream_type(arg_type):
            raise ValueError("Vitis axis interface can only be set on stream arguments")

        options = _collect_interface_options(
            {
                "register": register,
                "register_mode": register_mode,
                "depth": depth,
                "name": name,
                "bundle": bundle,
            },
            kwargs,
        )
        self._set_interface_pragma(index, "axis", options)

    def set_axilite(
        self,
        index: int,
        *,
        bundle: str | None = None,
        register: bool | None = None,
        clock: str | None = None,
        name: str | None = None,
        offset: str | None = None,
        storage_impl: AxiliteStorageImpl | None = None,
        **kwargs: str | int | bool | None,
    ):
        """
        Set the indexed argument or return value (index=-1) to be an AXI lite interface with the given options. For details of the options, please refer to [Vitis HLS Interface Pragma](https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-interface)

        In Vitis HLS, AXI lite interfaces can be applied to pointer (buffer) arguments and scalar return value. Setting an AXI lite interface on a pointer argument will cause the argument to be accessed through an AXI lite slave interface, which is typically used for control and configuration. Setting an AXI lite interface on the return value will cause it to be returned through an AXI lite slave interface, which is typically used for status reporting.
        """
        self._validate_interface_index(index, allow_return=True)
        if index == -1 and register is True:
            raise ValueError(
                "Vitis s_axilite return port does not support the register option"
            )

        options = _collect_interface_options(
            {
                "bundle": bundle,
                "register": register,
                "clock": clock,
                "name": name,
                "offset": offset,
                "storage_impl": storage_impl,
            },
            kwargs,
        )
        self._set_interface_pragma(index, "s_axilite", options)

    @terminate_on_error
    def run(
        self,
        mode: VitisMode,
        *args,
        exist_ok: bool = True,
        **kwargs,
    ) -> Any:
        if kwargs:
            raise TypeError(
                "Vitis backend only accepts positional kernel arguments and "
                "the exist_ok keyword"
            )
        if mode == "csim":
            return self.csim(*args, exist_ok=exist_ok)
        if args:
            raise TypeError("Vitis csyn does not accept runtime arguments")
        if mode == "csyn":
            return self.synth(exist_ok=exist_ok)
        raise NotImplementedError(f"Vitis mode '{mode}' is not implemented yet")

    @terminate_on_error
    def csim(self, *args, exist_ok: bool = True) -> Any:
        self._require_vitis_tool()
        project_path, cache_key = self._materialize_csim_cache(exist_ok=exist_ok)
        if not exist_ok:
            self.csimulator = None
            self._pcache_pop("vitis.csim", cache_key)
        simulator = self._get_csimulator(project_path, cache_key)
        return simulator.run(*args, exist_ok=exist_ok)

    @terminate_on_error
    def synth(self, *, exist_ok: bool = True) -> VitisSynthReport:
        """Generate an HLS csyn project and invoke Vitis HLS."""
        if not self.part:
            raise ValueError(
                "Vitis synthesis requires a part number; pass part=... (or "
                "device=...) to export('vitis', ...)."
            )
        self._require_vitis_tool()
        project_path = self.scaffold_project(exist_ok=exist_ok)
        self._invoke_csyn(project_path)
        artifacts = self._ensure_compiled()
        rpt = VitisSynthReport(project_path=project_path, top=artifacts.top)
        # automatically render
        rpt.render()
        return rpt

    @terminate_on_error
    def scaffold_project(
        self, project: str | None = None, *, exist_ok: bool = True
    ) -> Path:
        """
        Generate the HLS project files without invoking Vitis HLS.

        If the project argument is provided, the project will be generated to the specified path
        """
        if project is None and self._project_path is not None:
            return self._materialize_project(self._project_path, exist_ok=exist_ok)
        return self._materialize_project(project, exist_ok=exist_ok)

    def _materialize_project(
        self, project: Path | str | None = None, *, exist_ok: bool = True
    ) -> Path:
        project_path = make_project_path(
            project, f"allo-vitis-prj-{self.kernel.func_name}", exist_ok=exist_ok
        )

        artifacts = self._ensure_compiled()
        with stage(f"Generating Vitis HLS Project to: {project_path.resolve()}"):
            write_text_if_changed(project_path / "kernel.cpp", artifacts.kernel_cpp)
            write_text_if_changed(project_path / "kernel.h", artifacts.kernel_h)
            write_text_if_changed(
                project_path / "run.tcl",
                generate_run_tcl(artifacts.top, self.part, self.freq_mhz, self.flow),
            )

        self._project_path = project_path
        return project_path

    def _materialize_csim_cache(self, *, exist_ok: bool = True) -> tuple[Path, str]:
        artifacts = self._compile_for_csim()
        vitis_root = self._get_vitis_root()
        makefile = _generate_csim_makefile(
            vitis_root, self._get_csim_toolchain().template
        )
        payload = self._csim_cache_payload(artifacts, makefile)
        cache_key = stable_cache_hash(payload)
        project_path = self._cache_dir(
            "vitis",
            "csim",
            cache_key[:CSIM_CACHE_DIR_KEY_LENGTH],
        )
        cache_files = (
            project_path / "kernel.cpp",
            project_path / "kernel.h",
            project_path / CSIM_MAKEFILE,
            project_path / "cache.json",
        )

        if exist_ok and all(path.exists() for path in cache_files):
            return project_path, cache_key

        with stage(f"Generating Vitis C Simulation Cache to: {project_path}"):
            project_path.mkdir(parents=True, exist_ok=True)
            write_text_if_changed(project_path / "kernel.cpp", artifacts.kernel_cpp)
            write_text_if_changed(project_path / "kernel.h", artifacts.kernel_h)
            write_text_if_changed(project_path / CSIM_MAKEFILE, makefile)
            write_json_if_changed(
                project_path / "cache.json",
                {
                    "key": cache_key,
                    "payload": payload,
                    "exist_ok": exist_ok,
                },
            )

        return project_path, cache_key

    def _ensure_compiled(self) -> CompiledArtifacts:
        if self.artifacts is None:
            self.artifacts = self.compile()
        return self.artifacts

    def _invalidate_compiled_artifacts(self) -> None:
        self.artifacts = None
        self.csimulator = None

    def _get_working_module(self) -> ir.Module:
        """Return a fresh clone of the kernel module for in-place lowering.

        ``compile`` runs the HLS pipeline in place, so each invocation needs an
        independent copy; ``self.module`` stays pristine and recompilable after
        interface pragmas invalidate the cached artifacts.
        """
        return ir_ext.clone_module(self.module)

    @terminate_on_error
    def _validate_interface_index(
        self, index: int, *, allow_return: bool
    ) -> TypeBase | None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Vitis interface index must be an integer")

        arg_types = self.kernel.parse_argument_annotations()
        if index == -1:
            if not allow_return:
                raise ValueError("This Vitis interface cannot be set on return value")
            ret_types = self.kernel.parse_return_annotation()
            if len(ret_types) > 1:
                raise ValueError(
                    "Vitis backend supports at most one return value for "
                    "interface configuration"
                )
            if ret_types and isinstance(ret_types[0], ShapedType):
                raise ValueError(
                    "Vitis backend does not support shaped return interfaces. "
                    "Pass output buffers as explicit arguments instead."
                )
            return None

        if index < 0 or index >= len(arg_types):
            raise IndexError(
                f"Vitis interface index {index} is out of range for "
                f"{len(arg_types)} input arguments"
            )
        return arg_types[index]

    @terminate_on_error
    def _set_interface_pragma(
        self,
        index: int,
        mode: str,
        options: Mapping[str, Any],
    ) -> None:
        if mode not in _INTERFACE_MODES:
            raise ValueError(f"Unsupported Vitis HLS interface mode '{mode}'")
        self._interface_pragmas.setdefault(index, {})[mode] = InterfacePragma(
            mode=mode,
            options=_normalize_interface_options(mode, options),
        )
        self._invalidate_compiled_artifacts()

    def _validate_interface_pragmas(self) -> None:
        for index, pragmas in self._interface_pragmas.items():
            arg_type = self._validate_interface_index(
                index, allow_return="s_axilite" in pragmas
            )
            if "m_axi" in pragmas and not isinstance(arg_type, BufferType):
                raise ValueError(
                    "Vitis m_axi interface can only be set on buffer arguments"
                )
            if "axis" in pragmas and (
                arg_type is None or not _is_stream_type(arg_type)
            ):
                raise ValueError(
                    "Vitis axis interface can only be set on stream arguments"
                )

    def _get_vitis_env(self) -> dict[str, str]:
        if self.tool is not None:
            return dict(self.tool.env)
        with stage("Load Vitis environment"):
            sourced_env = _source_settings_env(self._settings64)
            if sourced_env is not None:
                return sourced_env
        return dict(os.environ)

    def _get_vitis_root(self) -> Path:
        if self._vitis_home is not None:
            return self._vitis_home
        if self._settings64.name == "settings64.sh":
            return self._settings64.parent
        if self.tool is not None and self.tool.executable.parent.name == "bin":
            return self.tool.executable.parent.parent
        return DEFAULT_VITIS_HOME

    def _get_csim_toolchain(self) -> CsimToolchain:
        """The probed C-simulation flavor (makefile template + version-discovered
        make vars). Native AMD-clang flow on Vitis 2025.2+, else the legacy gcc
        flow used by Vitis through 2024.2."""
        if self._csim_toolchain is None:
            self._csim_toolchain = discover_csim(self._get_vitis_root())
        return self._csim_toolchain

    def _interface_cache_payload(self) -> dict[int, dict[str, Mapping[str, Any]]]:
        return {
            index: {
                mode: pragma.options
                for mode, pragma in sorted(pragmas.items(), key=lambda item: item[0])
            }
            for index, pragmas in sorted(self._interface_pragmas.items())
        }

    def _csim_cache_payload(
        self, artifacts: CompiledArtifacts, makefile: str
    ) -> dict[str, Any]:
        # The compiled .so is fully determined by the emitted C++, the makefile
        # recipe (incl. OPT_FLAGS) and flavor, and the probed clang/gcc toolchain.
        # kernel_h is derived from kernel_cpp, and the upstream MLIR module is
        # already captured by kernel_cpp, so neither is a separate field; the
        # vitis_hls tool version is irrelevant to a compiler-built .so.
        toolchain = self._get_csim_toolchain()
        return {
            "phase": "python-native-csim",
            "flavor": toolchain.flavor,
            "kernel_cpp_sha256": text_hash(artifacts.kernel_cpp),
            "makefile_sha256": text_hash(makefile),
            "toolchain": dict(sorted(toolchain.make_vars.items())),
            "overrides": dict(sorted(self._csim_make_vars.items())),
        }

    def _get_csimulator(
        self, project_path: Path, cache_key: str
    ) -> PythonNativeCSimulator:
        if self.csimulator is not None and self.csimulator.project_path == project_path:
            return self.csimulator
        cached = self._pcache_get("vitis.csim", cache_key)
        if cached:
            self.csimulator = cached
            return cached
        toolchain = self._get_csim_toolchain()
        self.csimulator = PythonNativeCSimulator(
            top=self.kernel.func_name,
            project_path=project_path,
            vitis_root=self._get_vitis_root(),
            env=self._get_vitis_env(),
            arg_types=self.kernel.parse_argument_annotations(),
            res_types=self.kernel.parse_return_annotation(),
            # Probed toolchain paths first; explicit user overrides win.
            make_vars={**toolchain.make_vars, **self._csim_make_vars},
            makefile_template=toolchain.template,
        )
        self._pcache_set("vitis.csim", cache_key, self.csimulator)
        return self.csimulator

    def _invoke_csyn(self, project_path: Path) -> None:
        log_path = _synth_log_path(project_path)
        if self.tool.name == "vitis-run":
            cmd = [
                os.fspath(self.tool.executable),
                "--mode",
                "hls",
                "--tcl",
                "--work_dir",
                ".",
                "run.tcl",
            ]
        elif self.tool.name == "vitis_hls":
            cmd = [os.fspath(self.tool.executable), "-f", "run.tcl"]
        else:
            assert False, "Unknown Vitis tool detected: " + self.tool.name

        with stage(
            "Running Vitis HLS Synthesis",
            on_error=lambda error: _log_synth_failure(log_path, error),
            on_exit=lambda: _log_synth_note(log_path),
        ):
            run_command(cmd, cwd=project_path, env=dict(self.tool.env))

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

    @terminate_on_error
    def compile(self) -> CompiledArtifacts:
        if self.kernel.func_name == "kernel":
            raise ValueError(
                "'kernel' is a reserved name for Vitis HLS. Please rename your kernel function."
            )
        self._validate_top_abi()
        self._validate_interface_pragmas()

        # No cached artifacts are used for HLS codegen now,
        # because the codegen is typically much faster than sim/synth/impl
        with stage("Compiling Vitis HLS Kernels"):
            artifacts = self._emit_artifacts(apint_wrapper=False)
            self.artifacts = artifacts
            return artifacts

    def _compile_for_csim(self) -> CompiledArtifacts:
        """Codegen for C simulation: wrap any non-standard-width APInt boundary
        with a std-width interface so ctypes can call the top. The synthesizable
        interface (``compile``/``hls_code``/``synth``) keeps the real ap_int."""
        if self.kernel.func_name == "kernel":
            raise ValueError(
                "'kernel' is a reserved name for Vitis HLS. Please rename your kernel function."
            )
        self._validate_top_abi()
        self._validate_interface_pragmas()
        with stage("Compiling Vitis HLS Kernels (csim)"):
            return self._emit_artifacts(apint_wrapper=True)

    def _emit_artifacts(self, *, apint_wrapper: bool) -> CompiledArtifacts:
        module = self._get_working_module()
        if lookup_kernel(module, self.kernel.func_name) is None:
            raise RuntimeError(
                f"Kernel function {self.kernel.func_name} not found in the module"
            )
        if apint_wrapper:
            run_pipeline(
                module,
                "builtin.module(materialize-apint-wrapper{"
                f"top={self.kernel.func_name}}})",
            )
        run_pipeline(module, HLS_PREPARE_PIPELINE)
        hls_code = emit_vivado_hls(module)
        if hls_code is None:
            raise RuntimeError("Failed to emit Vitis HLS code")
        hls_code = _add_extern_c_to_top(hls_code, self.kernel.func_name)
        hls_code = _apply_interface_pragmas(
            hls_code, self.kernel.func_name, self._interface_pragmas
        )
        return CompiledArtifacts(
            kernel_cpp=hls_code,
            kernel_h=generate_kernel_header(hls_code, self.kernel.func_name),
            top=self.kernel.func_name,
        )
