import os
import subprocess

from pathlib import Path

from .utils import make_project_path
from .base import Backend
from dataclasses import dataclass
from typing import Any, Literal

from .._C.ir import ModuleOp, UnitAttr
from .._C.passes import emit_vivado_hls
from ..lang.kernel import Kernel

VitisMode = Literal["csim", "csyn", "sw_emu", "hw_emu", "hw"]
FlowTarget = Literal["vitis", "vivado"]


@dataclass(frozen=True)
class CompiledArtifacts:
    kernel_cpp: str
    kernel_h: str
    top: str
    module: ModuleOp


def _generate_project_tcl(
    top: str, part: str, freq_mhz: float, flow_target: FlowTarget
) -> str:
    clock_period = 1000.0 / freq_mhz  # Convert MHz to ns

    return f"""# Auto-generated project.tcl for Vitis HLS

    open_project -reset hls_prj
    set_top {top}

    add_files kernel.cpp
    open_solution -reset solution1 -flow_target {flow_target}
    set_part {{{part}}}
    create_clock -period {clock_period:.4f} -name default
    """


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


class VitisSynthesisReport:
    def render(self) -> None:
        pass


class Vitis(Backend):
    name = "vitis"

    def __init__(self, kernel: Kernel, vitis_home: str = "", project_path: str = ""):
        vitis_env = os.getenv("VITIS_HOME", "")
        if not vitis_env and not vitis_home:
            raise RuntimeError(
                "Don't know where to find vitis. Please source Vitis settings64.sh or explicitly provide the path to Vitis installation via vitis_home argument."
            )
        if not vitis_home:
            vitis_home = vitis_env
        self._vitis_home = self._verify_vitis_home(vitis_home)
        super().__init__(kernel)
        self._project_path = self.scaffold_project(project_path)
        self._part: str = ""
        self._platform: str = ""
        self._freq_mhz: float = 300.0
        self._flow: FlowTarget = "vitis"
        self.artifacts: CompiledArtifacts | None = None

    @staticmethod
    def _verify_vitis_home(vitis_home: str) -> Path:
        vitis_path = Path(vitis_home)
        if not (vitis_path / "bin" / "vitis").exists():
            raise RuntimeError(
                f"Vitis executable not found in {vitis_home}. Please check your VITIS_HOME environment variable or the provided vitis_home argument."
            )
        return vitis_path

    @property
    def part(self) -> str:
        return self._part

    @part.setter
    def part(self, part: str) -> None:
        if self._platform:
            raise RuntimeError("Cannot set part after platform is set")
        self._part = part

    @property
    def platform(self) -> str:
        return self._platform

    @platform.setter
    def platform(self, platform: str) -> None:
        if self._part:
            raise RuntimeError("Cannot set platform after part is set")
        self._platform = platform
        self._part = PART_NUMBERS.get(platform, "")
        if not self._part:
            raise ValueError(
                f"Unknown platform {platform}. Please set part number manually."
            )

    @property
    def freq_mhz(self) -> float:
        return self._freq_mhz

    @freq_mhz.setter
    def freq_mhz(self, freq: float) -> None:
        if freq <= 0:
            raise ValueError("Frequency must be positive")
        self._freq_mhz = freq

    @property
    def flow(self) -> FlowTarget:
        return self._flow

    @flow.setter
    def flow(self, flow: FlowTarget) -> None:
        if flow not in ["vitis", "vivado"]:
            raise ValueError("Flow must be either 'vitis' or 'vivado'")
        self._flow = flow

    def run(self, mode: VitisMode, *args, **kwargs) -> Any:
        if mode == "csim":
            pass
        if mode == "csyn":
            return self.synth()

    def synth(self):
        artifacts = self._ensure_compiled()
        run_tcl = f"""# Auto-generated synth.tcl for Vitis HLS

        source {self._project_path}/project.tcl

        csynth_design
        {f"export_design -format ip_catalog artifacts/{artifacts.top}.zip"
            if self.flow == "vivado" else f"export_design -format xo artifacts/{artifacts.top}.xo"}
        """

        (self._project_path / "synth.tcl").write_text(run_tcl)
        work_dir = self._project_path / "synth"
        try:
            subprocess.run(
                [
                    self._vitis_home / "bin" / "vitis-run",
                    "--work_dir",
                    work_dir,
                    "--mode",
                    "hls",
                    "--tcl",
                    self._project_path / "synth.tcl",
                ],
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Vitis HLS synthesis failed")

        print(f"\nVitis HLS synthesis completed successfully.")

        return self._parse_synth_report(work_dir)

    def _parse_synth_report(self, work_dir: Path) -> VitisSynthesisReport:
        pass

    def scaffold_project(
        self, project: str | None = None, *, overwrite: bool = False
    ) -> Path:
        project_path = make_project_path(
            project, f"allo-vitis-prj-{self.kernel.func_name}", overwrite
        )
        (project_path / "artifacts").mkdir(exist_ok=True)
        (project_path / "synth").mkdir(exist_ok=True)

        artifacts = self._ensure_compiled()
        (project_path / "kernel.cpp").write_text(artifacts.kernel_cpp)
        (project_path / "kernel.h").write_text(artifacts.kernel_h)
        (project_path / "project.tcl").write_text(
            _generate_project_tcl(artifacts.top, self.part, self.freq_mhz, self.flow)
        )

        self._project_path = project_path
        return project_path

    def _ensure_compiled(self) -> CompiledArtifacts:
        if self.artifacts is None:
            self.artifacts = self.compile()
        return self.artifacts

    def compile(self) -> CompiledArtifacts:
        if self.kernel.func_name == "kernel":
            raise ValueError(
                "'kernel' is a reserved name for Vitis HLS. Please rename your kernel function."
            )

        module = self._get_working_module()
        top_fn = module.lookup_func(self.kernel.func_name)
        if top_fn is None:
            raise RuntimeError(
                f"Kernel function {self.kernel.func_name} not found in the module"
            )
        top_fn.set_attr("top", UnitAttr.get(module.get_context()))

        hls_code = emit_vivado_hls(module)
        hls_header = _extract_kernel_header(hls_code, self.kernel.func_name)
        hls_code = _postprocess_hls_code(hls_code)
        artifacts = CompiledArtifacts(
            kernel_cpp=hls_code,
            kernel_h=hls_header,
            top=self.kernel.func_name,
            module=module,
        )
        self.artifacts = artifacts
        return artifacts
