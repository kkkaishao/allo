from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...logging import (
    CommandError,
    completed_output,
    log_info,
    log_tail,
    read_text_tail,
    stage,
)

SYNTH_LOG = Path("logs") / "hls_run_tcl.log"
SYNTH_LOG_TAIL_LINES = 100
TEMPLATE_DIR = Path(__file__).with_name("templates")

_INTERFACE_MODES = ("m_axi", "axis", "s_axilite")
_AXI_OFFSET_VALUES = {"off", "direct", "slave"}
_AXIS_REGISTER_MODE_VALUES = {"forward", "reverse", "both", "off"}
_AXILITE_STORAGE_IMPL_VALUES = {"auto", "bram", "uram"}
_INTERFACE_OPTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_AXI_OPTION_ORDER = (
    "offset",
    "bundle",
    "channel",
    "depth",
    "latency",
    "num_read_outstanding",
    "num_write_outstanding",
    "max_read_burst_length",
    "max_write_burst_length",
    "max_widen_bitwidth",
    "alignment_byte_size",
    "name",
)
_AXIS_OPTION_ORDER = ("register", "register_mode", "depth", "name", "bundle")
_AXILITE_OPTION_ORDER = (
    "bundle",
    "register",
    "clock",
    "name",
    "offset",
    "storage_impl",
)
_VITIS_VERSION_RE = re.compile(r"\bv?(\d{4}\.\d+(?:\.\d+)?)\b")


@dataclass(frozen=True)
class VitisTool:
    name: str
    executable: Path
    env: Mapping[str, str]
    version: str = "unknown"


def _render_template(name: str, **kwargs) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8").format(**kwargs)


def _generate_run_tcl(top: str, part: str, freq_mhz: float, flow_target: str) -> str:
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


def _split_cpp_arguments(arguments: str) -> list[str]:
    parts = []
    start = 0
    angle_depth = 0
    bracket_depth = 0
    for i, char in enumerate(arguments):
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif char == "," and angle_depth == 0 and bracket_depth == 0:
            parts.append(arguments[start:i].strip())
            start = i + 1
    tail = arguments[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_cpp_arg_name(argument: str) -> str:
    arg = argument.strip()
    while arg.endswith("]"):
        arg = re.sub(r"\s*\[[^\]]*\]\s*$", "", arg).rstrip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", arg)
    if match is None:
        raise RuntimeError(f"Failed to parse C++ argument name from '{argument}'")
    return match.group(1)


def _extract_top_port_names(hls_code: str, top: str) -> list[str]:
    marker = _top_signature_marker(top)
    for line in hls_code.splitlines():
        stripped = line.strip()
        if marker not in stripped or not stripped.endswith("{"):
            continue
        args_begin = stripped.find(marker) + len(marker)
        args_end = stripped.rfind(")")
        if args_end < args_begin:
            raise RuntimeError(f"Failed to parse emitted definition for {top}")
        arguments = stripped[args_begin:args_end].strip()
        if not arguments:
            return []
        return [_extract_cpp_arg_name(arg) for arg in _split_cpp_arguments(arguments)]
    raise RuntimeError(f"Failed to find emitted definition for top function {top}")


def _set_if_provided(options: dict[str, Any], name: str, value: object) -> None:
    if value is not None:
        options[name] = value


def _validate_interface_option_name(name: str) -> None:
    if not _INTERFACE_OPTION_RE.match(name):
        raise ValueError(f"Invalid Vitis HLS interface option name '{name}'")


def _validate_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Vitis HLS interface option '{name}' must be a string")


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Vitis HLS interface option '{name}' must be a positive integer"
        )


def _validate_optional_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"Vitis HLS interface option '{name}' must be a boolean")


def _is_stream_type(arg_type: object) -> bool:
    return arg_type.__class__.__name__ == "StreamType"


def _normalize_interface_options(
    mode: str,
    options: Mapping[str, Any],
) -> dict[str, str | int | bool | None]:
    normalized: dict[str, str | int | bool | None] = {}
    for name, value in options.items():
        _validate_interface_option_name(name)
        if value is None:
            continue
        if name == "register":
            _validate_optional_bool(name, value)
            normalized[name] = value
            continue
        if name in {
            "depth",
            "latency",
            "num_read_outstanding",
            "num_write_outstanding",
            "max_read_burst_length",
            "max_write_burst_length",
            "max_widen_bitwidth",
            "alignment_byte_size",
        }:
            _validate_positive_int(name, value)
            normalized[name] = value
            continue
        if name == "offset" and mode == "m_axi":
            _validate_non_empty_string(name, value)
            if value not in _AXI_OFFSET_VALUES:
                raise ValueError(
                    "Vitis HLS m_axi offset must be one of: off, direct, slave"
                )
            normalized[name] = value
            continue
        if name in {"bundle", "channel", "name", "clock", "offset"}:
            _validate_non_empty_string(name, value)
            normalized[name] = value
            continue
        if name == "register_mode":
            _validate_non_empty_string(name, value)
            if value not in _AXIS_REGISTER_MODE_VALUES:
                raise ValueError(
                    "Vitis HLS axis register_mode must be one of: "
                    "forward, reverse, both, off"
                )
            normalized[name] = value
            continue
        if name == "storage_impl":
            _validate_non_empty_string(name, value)
            if value not in _AXILITE_STORAGE_IMPL_VALUES:
                raise ValueError(
                    "Vitis HLS s_axilite storage_impl must be one of: "
                    "auto, bram, uram"
                )
            normalized[name] = value
            continue
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, bool):
            normalized[name] = value
        elif isinstance(value, int):
            normalized[name] = value
        elif isinstance(value, str):
            if not value:
                raise ValueError(
                    f"Vitis HLS interface option '{name}' must not be empty"
                )
            normalized[name] = value
        else:
            raise TypeError(
                f"Unsupported Vitis HLS interface option value for '{name}': "
                f"{type(value).__name__}"
            )
    return normalized


def _interface_option_order(mode: str) -> tuple[str, ...]:
    if mode == "m_axi":
        return _AXI_OPTION_ORDER
    if mode == "axis":
        return _AXIS_OPTION_ORDER
    if mode == "s_axilite":
        return _AXILITE_OPTION_ORDER
    raise ValueError(f"Unsupported Vitis HLS interface mode '{mode}'")


def _render_interface_options(
    options: Mapping[str, str | int | bool | None],
    order: tuple[str, ...],
) -> list[str]:
    rendered = []
    remaining = dict(options)
    for name in order:
        if name in remaining:
            value = remaining.pop(name)
            if value is True:
                rendered.append(name)
            elif value not in (False, None):
                rendered.append(f"{name}={value}")
    for name in sorted(remaining):
        value = remaining[name]
        if value is True:
            rendered.append(name)
        elif value not in (False, None):
            rendered.append(f"{name}={value}")
    return rendered


def _render_interface_pragma(pragma: Any, port: str) -> str:
    options = _render_interface_options(
        pragma.options, _interface_option_order(pragma.mode)
    )
    suffix = " " + " ".join(options) if options else ""
    return f"#pragma HLS INTERFACE mode={pragma.mode} port={port}{suffix}"


def _apply_interface_pragmas(
    hls_code: str,
    top: str,
    pragmas: Mapping[int, Mapping[str, Any]],
) -> str:
    if not pragmas:
        return hls_code

    ports = _extract_top_port_names(hls_code, top)
    lines = []
    inserted = False
    marker = _top_signature_marker(top)
    mode_order = {mode: i for i, mode in enumerate(_INTERFACE_MODES)}
    index_order = sorted(pragmas, key=lambda index: (index == -1, index))
    for line in hls_code.splitlines():
        lines.append(line)
        stripped = line.strip()
        if inserted or marker not in stripped or not stripped.endswith("{"):
            continue
        indent = line[: len(line) - len(line.lstrip())] + "  "
        for index in index_order:
            port = "return" if index == -1 else ports[index]
            for mode, pragma in sorted(
                pragmas[index].items(), key=lambda item: mode_order[item[0]]
            ):
                lines.append(indent + _render_interface_pragma(pragma, port))
        inserted = True
    if not inserted:
        raise RuntimeError(f"Failed to insert interface pragmas for {top}")
    return "\n".join(lines) + ("\n" if hls_code.endswith("\n") else "")


def _synth_log_path(project_path: Path) -> Path:
    return project_path / SYNTH_LOG


def _log_synth_note(log_path: Path) -> None:
    log_info(f"Synthesis log exported to: {log_path}")


def _log_synth_failure(log_path: Path, error: Exception) -> None:
    tail = read_text_tail(log_path, max_lines=SYNTH_LOG_TAIL_LINES)
    if not tail and isinstance(error, CommandError):
        tail = error.output_tail(SYNTH_LOG_TAIL_LINES)
    log_tail("Synthesis log tail", tail, max_lines=SYNTH_LOG_TAIL_LINES)


def _settings_from_vitis_home(vitis_home: str | None) -> Path | None:
    if not vitis_home:
        return None
    return Path(vitis_home) / "settings64.sh"


def _resolve_settings64(
    settings64: str | os.PathLike[str] | None,
    vitis_home: str | None,
    default_settings: Path,
) -> Path:
    if settings64:
        return Path(settings64)
    env_settings = os.getenv("VITIS_SETTINGS64")
    if env_settings:
        return Path(env_settings)
    home_settings = _settings_from_vitis_home(vitis_home or os.getenv("VITIS_HOME"))
    if home_settings:
        return home_settings
    return default_settings


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


def _version_commands(tool_name: str) -> tuple[str, ...]:
    if tool_name == "vitis-run":
        return ("--version",)
    return ("-version", "--version")


def _parse_vitis_version(output: str) -> str:
    match = _VITIS_VERSION_RE.search(output)
    return match.group(1) if match is not None else "unknown"


def _probe_vitis_version(
    executable: Path, tool_name: str, env: Mapping[str, str]
) -> str:
    for version_arg in _version_commands(tool_name):
        result = subprocess.run(
            [os.fspath(executable), version_arg],
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = completed_output(result)
        if output:
            version = _parse_vitis_version(output)
            if version != "unknown":
                return version
    return "unknown"


def _find_tool_in_env(env: Mapping[str, str]) -> VitisTool | None:
    path = env.get("PATH", "")
    for tool_name in ("vitis-run", "vitis_hls"):
        executable = shutil.which(tool_name, path=path)
        if executable:
            tool_path = Path(executable)
            return VitisTool(
                tool_name,
                tool_path,
                dict(env),
                _probe_vitis_version(tool_path, tool_name, env),
            )
    return None


def _probe_vitis_tool(settings64: Path) -> VitisTool | None:
    sourced_env = _source_settings_env(settings64)
    if sourced_env is not None:
        tool = _find_tool_in_env(sourced_env)
        if tool is not None:
            return tool

    return _find_tool_in_env(os.environ)


def _detect_vitis_tool(settings64: Path) -> VitisTool | None:
    with stage("Detecting Vitis HLS Toolchain"):
        tool = _probe_vitis_tool(settings64)
    if tool is not None:
        log_info(f"Using Vitis {tool.executable}, Version: {tool.version}")
    return tool
