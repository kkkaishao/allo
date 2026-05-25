from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Generic, TypeVar, ParamSpec

from ..base import Backend, text_hash, write_json_if_changed, write_text_if_changed
from ..utils import make_project_path
from .csim import (
    CSIM_MAKEFILE,
    PythonNativeCSimulator,
    _CSIM_UNSET,
    _generate_csim_makefile,
    _normalize_csim_make_vars,
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
    _set_if_provided,
    _source_settings_env,
    _synth_log_path,
)
from ..._C import passes
from ..._C.ir import UnitAttr
from ..._C.passes import emit_vivado_hls
from ...lang.core import BufferType, ShapedType, TypeBase
from ...lang.kernel import Kernel
from ...logging import log_warning, run_command, stage, terminate_on_error

VitisMode = Literal["csim", "csyn", "sw_emu", "hw_emu", "hw"]
FlowTarget = Literal["vitis", "vivado"]
AxiOffset = Literal["off", "direct", "slave"]
AxisRegisterMode = Literal["forward", "reverse", "both", "off"]
AxiliteStorageImpl = Literal["auto", "bram", "uram"]

DEFAULT_DEVICE = "u280"
DEFAULT_FREQ_MHZ = 300.0
DEFAULT_VITIS_SETTINGS = Path("/opt/xilinx/2025.2/Vitis/settings64.sh")
HLS_PREPARE_PIPELINE = """
builtin.module(convert-allo-to-func,func.func(convert-linalg-to-affine-loops),canonicalize,cse)
"""
CSIM_CACHE_DIR_KEY_LENGTH = 24


@dataclass(frozen=True)
class CompiledArtifacts:
    kernel_cpp: str
    kernel_h: str
    top: str


@dataclass(frozen=True)
class InterfacePragma:
    mode: str
    options: Mapping[str, str | int | bool | None]


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

P = ParamSpec("P")
R = TypeVar("R")


class Vitis(Backend, Generic[P, R]):
    name = "vitis"

    @terminate_on_error
    def __init__(
        self,
        kernel: Kernel[P, R] | None = None,
        vitis_home: str | None = None,
        project_path: str | None = None,
        *,
        device: str | None = None,
        part: str | None = None,
        freq_mhz: float = 300.0,
        flow: FlowTarget = "vitis",
    ):
        super().__init__(kernel)
        # setup toolchain paths
        self._settings64 = (
            Path(vitis_home) / "settings64.sh" if vitis_home else DEFAULT_VITIS_SETTINGS
        )
        self._vitis_home = Path(vitis_home) if vitis_home else None
        self.tool: VitisTool | None = None
        # setup project related settings
        self._project_path = Path(project_path) if project_path else None
        self._freq_mhz = freq_mhz
        self._flow: FlowTarget = flow
        self._csim_make_vars: dict[str, str] = {}
        self._init_part_and_device(part, device)
        # interface pragmas set by the user
        self._interface_pragmas: dict[int, dict[str, InterfacePragma]] = {}
        # intermediate stuffs
        self.artifacts: CompiledArtifacts | None = None
        self.csimulator: PythonNativeCSimulator | None = None

    def _init_part_and_device(self, part: str | None, device: str | None) -> None:
        if part and device:
            if PART_NUMBERS.get(device) != part:
                raise ValueError("Cannot specify both part number and device")
            self._part = part
            self._device = device
        if part is None and device is None:
            log_warning(
                "Neither device nor part number is specified for Vitis backend. The backend can be only used for C simulation until the part number is set, which is required for synthesis and implementation."
            )
            self._part = ""
            self._device = ""
        elif device is not None:
            part = PART_NUMBERS.get(device)
            if not part:
                raise ValueError(
                    f"Unknown device {device}. Please specify part number directly."
                )
            self._device = device
            self._part = part
        elif part is not None:
            self._part = part
            self._device = ""

    def _require_vitis_tool(self):
        if self.tool is None:
            self.tool = detect_vitis_tool(self._settings64)

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

    def call_kernel(self, kernel: Kernel, *args: P.args, **kwargs: P.kwargs) -> R:
        backend = Vitis(
            kernel,
            os.fspath(self._vitis_home) if self._vitis_home is not None else None,
        )
        backend._csim_make_vars = self._csim_make_vars.copy()
        backend.tool = self.tool
        try:
            return backend.csim(*args, **kwargs)
        finally:
            self.tool = backend.tool

    @terminate_on_error
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.kernel is None:
            raise RuntimeError("Vitis backend is not bound to a kernel")
        return self.csim(*args, **kwargs)

    def set_csim_override(
        self,
        *,
        vitis_root: str = _CSIM_UNSET,
        cxx: str = _CSIM_UNSET,
        gcc_toolchain: str = _CSIM_UNSET,
        vitis_host_lib: str = _CSIM_UNSET,
        mathhls_lib: str = _CSIM_UNSET,
        fpo_lib: str = _CSIM_UNSET,
        kernel_cpp: str = _CSIM_UNSET,
        kernel_h: str = _CSIM_UNSET,
        out: str = _CSIM_UNSET,
        hls_includes: str = _CSIM_UNSET,
        hls_defines: str = _CSIM_UNSET,
        hls_cxxflags: str = _CSIM_UNSET,
        hls_ldflags: str = _CSIM_UNSET,
        extra_cxxflags: str = _CSIM_UNSET,
        extra_ldflags: str = _CSIM_UNSET,
        **kwargs: str,
    ):
        updates: dict[str, str] = {}
        for key, value in {
            "vitis_root": vitis_root,
            "cxx": cxx,
            "gcc_toolchain": gcc_toolchain,
            "vitis_host_lib": vitis_host_lib,
            "mathhls_lib": mathhls_lib,
            "fpo_lib": fpo_lib,
            "kernel_cpp": kernel_cpp,
            "kernel_h": kernel_h,
            "out": out,
            "hls_includes": hls_includes,
            "hls_defines": hls_defines,
            "hls_cxxflags": hls_cxxflags,
            "hls_ldflags": hls_ldflags,
            "extra_cxxflags": extra_cxxflags,
            "extra_ldflags": extra_ldflags,
        }.items():
            if value is not _CSIM_UNSET:
                updates[key] = value
        updates.update(kwargs)

        for key, value in _normalize_csim_make_vars(updates).items():
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

        options: dict[str, Any] = {}
        for key, value in {
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
        }.items():
            _set_if_provided(options, key, value)
        options.update(kwargs)
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
        Set the indexed argument to be an AXI stream interface with the given options. For details of the options, please refer to [Vitis HLS Interface Pragma](https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-interface)

        The API cannot be used until frontend Stream type support is enabled, which is expected to be available in a future release.
        """
        raise NotImplementedError(
            "Vitis axis interface is not available until frontend Stream type "
            "support is enabled."
        )
        arg_type = self._validate_interface_index(index, allow_return=False)
        if not _is_stream_type(arg_type):
            raise ValueError("Vitis axis interface can only be set on stream arguments")

        options: dict[str, Any] = {}
        for key, value in {
            "register": register,
            "register_mode": register_mode,
            "depth": depth,
            "name": name,
            "bundle": bundle,
        }.items():
            _set_if_provided(options, key, value)
        options.update(kwargs)
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

        options: dict[str, Any] = {}
        for key, value in {
            "bundle": bundle,
            "register": register,
            "clock": clock,
            "name": name,
            "offset": offset,
            "storage_impl": storage_impl,
        }.items():
            _set_if_provided(options, key, value)
        options.update(kwargs)
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
            self._process_cache_pop("vitis.csim", cache_key)
        simulator = self._get_csimulator(project_path, cache_key)
        return simulator.run(*args, exist_ok=exist_ok)

    @terminate_on_error
    def synth(self, *, exist_ok: bool = True) -> VitisSynthReport:
        """Generate an HLS csyn project and invoke Vitis HLS."""
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
        artifacts = self._ensure_compiled()
        vitis_root = self._get_vitis_root()
        makefile = _generate_csim_makefile(vitis_root)
        payload = self._csim_cache_payload(artifacts, makefile)
        cache_key = self._cache_key(payload)
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

    # def _release_working_module(self) -> None:
    #     self.module = None
    #     self._module_owner = None

    def _invalidate_compiled_artifacts(self) -> None:
        self.artifacts = None
        self.csimulator = None

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
        return DEFAULT_VITIS_SETTINGS.parent

    def _tool_cache_payload(self) -> dict[str, str]:
        return {
            "name": self.tool.name,
            "executable": os.fspath(self.tool.executable),
            "version": self.tool.version,
            "settings64": os.fspath(self._settings64),
        }

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
        return {
            "backend": self.name,
            "phase": "python-native-csim",
            "top": artifacts.top,
            "kernel_cpp_sha256": text_hash(artifacts.kernel_cpp),
            "kernel_h_sha256": text_hash(artifacts.kernel_h),
            "makefile_sha256": text_hash(makefile),
            "tool": self._tool_cache_payload(),
        }

    def _get_csimulator(
        self, project_path: Path, cache_key: str
    ) -> PythonNativeCSimulator:
        if self.csimulator is not None and self.csimulator.project_path == project_path:
            return self.csimulator
        cached = self._process_cache_get("vitis.csim", cache_key)
        if cached:
            self.csimulator = cached
            return cached
        self.csimulator = PythonNativeCSimulator(
            top=self.kernel.func_name,
            project_path=project_path,
            vitis_root=self._get_vitis_root(),
            env=self._get_vitis_env(),
            arg_types=self.kernel.parse_argument_annotations(),
            res_types=self.kernel.parse_return_annotation(),
            make_vars=self._csim_make_vars,
        )
        self._process_cache_set("vitis.csim", cache_key, self.csimulator)
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
            module = self._get_working_module()
            top_fn = module.lookup_kernel(self.kernel.func_name)
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
            hls_code = _apply_interface_pragmas(
                hls_code, self.kernel.func_name, self._interface_pragmas
            )
            # top_fn = None
            # module = None
            # self._release_working_module()

            artifacts = CompiledArtifacts(
                kernel_cpp=hls_code,
                kernel_h=generate_kernel_header(hls_code, self.kernel.func_name),
                top=self.kernel.func_name,
            )
            self.artifacts = artifacts
            return artifacts
