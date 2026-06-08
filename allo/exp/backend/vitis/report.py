# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import xml.etree.ElementTree as ET

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ...logging import log_table, terminate_on_error

CSYNTH_XML = Path("hls_prj") / "solution1" / "syn" / "report" / "csynth.xml"
MISSING = "-"
RESOURCE_FIELDS = ("BRAM_18K", "DSP", "FF", "LUT", "URAM")


@dataclass(frozen=True)
class VitisSynthReport:
    project_path: Path
    top: str | None = None

    @property
    def xml_path(self) -> Path:
        return self.project_path / CSYNTH_XML

    @terminate_on_error
    def render(self) -> None:
        root = _read_xml(self.xml_path)
        log_table(
            "Vitis Synthesis Summary",
            ("Metric", "Value"),
            _summary_rows(root, self.top),
        )
        _render_table(
            "Scheduling",
            ("Metric", "Value"),
            _scheduling_rows(root),
        )
        _render_table(
            "Loop Latency",
            ("Loop", "Trip Count", "Latency", "II", "Depth", "Slack"),
            _loop_rows(root),
        )
        _render_table(
            "Resources",
            ("Resource", "Used", "Available", "Utilization"),
            _resource_rows(root),
        )
        mapping = _sw_to_hw_mapping(root)
        if mapping is not None:
            columns, rows = mapping
            _render_table("SW-to-HW Mapping", columns, rows)


def _read_xml(path: Path) -> ET.Element:
    if not path.exists():
        raise FileNotFoundError(f"Vitis synthesis report not found: {path}")
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as error:
        raise RuntimeError(f"Failed to parse Vitis synthesis report {path}: {error}")


def _text(node: ET.Element | None, path: str, default: str = MISSING) -> str:
    if node is None:
        return default
    child = node.find(path)
    if child is None or child.text is None:
        return default
    text = child.text.strip()
    return text if text else default


def _with_unit(value: str, unit: str) -> str:
    if value == MISSING or unit == MISSING:
        return value
    return f"{value} {unit}"


def _summary_rows(root: ET.Element, top: str | None) -> list[tuple[str, str]]:
    timing = root.find("./PerformanceEstimates/SummaryOfTimingAnalysis")
    target_unit = _text(root, "./UserAssignments/unit")
    timing_unit = _text(timing, "unit", target_unit)
    return [
        ("Top", top or _text(root, "./UserAssignments/TopModelName")),
        ("Vitis Version", _text(root, "./ReportVersion/Version")),
        ("Part", _text(root, "./UserAssignments/Part")),
        (
            "Target Clock",
            _with_unit(_text(root, "./UserAssignments/TargetClockPeriod"), target_unit),
        ),
        (
            "Estimated Clock",
            _with_unit(_text(timing, "EstimatedClockPeriod"), timing_unit),
        ),
    ]


def _scheduling_rows(root: ET.Element) -> list[tuple[str, str]]:
    latency = root.find("./PerformanceEstimates/SummaryOfOverallLatency")
    unit = _text(latency, "unit", "clock cycles")
    interval = _interval(latency)
    rows = [
        ("Best Latency", _with_unit(_text(latency, "Best-caseLatency"), unit)),
        ("Average Latency", _with_unit(_text(latency, "Average-caseLatency"), unit)),
        ("Worst Latency", _with_unit(_text(latency, "Worst-caseLatency"), unit)),
        ("Best Real Time", _text(latency, "Best-caseRealTimeLatency")),
        ("Average Real Time", _text(latency, "Average-caseRealTimeLatency")),
        ("Worst Real Time", _text(latency, "Worst-caseRealTimeLatency")),
    ]
    if interval != MISSING:
        rows.append(("Interval", interval))
    return rows


def _interval(latency: ET.Element | None) -> str:
    if latency is None:
        return MISSING
    minimum = _text(latency, "Interval-min")
    maximum = _text(latency, "Interval-max")
    if minimum != MISSING and maximum != MISSING:
        return minimum if minimum == maximum else f"{minimum} - {maximum}"
    return _text(latency, "PipelineInitiationInterval")


def _loop_rows(root: ET.Element) -> list[tuple[str, str, str, str, str, str]]:
    loops = root.find("./PerformanceEstimates/SummaryOfLoopLatency")
    if loops is None:
        return []
    rows = []
    for loop in loops:
        rows.append(
            (
                _text(loop, "Name", loop.tag),
                _text(loop, "TripCount"),
                _text(loop, "Latency"),
                _text(loop, "PipelineII"),
                _text(loop, "PipelineDepth"),
                _text(loop, "Slack"),
            )
        )
    return rows


def _resource_rows(root: ET.Element) -> list[tuple[str, str, str, str]]:
    resources = root.find("./AreaEstimates/Resources")
    available = root.find("./AreaEstimates/AvailableResources")
    rows = []
    for name in RESOURCE_FIELDS:
        used = _text(resources, name)
        total = _text(available, name)
        rows.append((name, used, total, _utilization(used, total)))
    return rows


def _utilization(used: str, total: str) -> str:
    try:
        used_value = float(used)
        total_value = float(total)
    except ValueError:
        return MISSING
    if total_value <= 0:
        return MISSING
    return f"{used_value / total_value * 100:.1f}%"


def _sw_to_hw_mapping(
    root: ET.Element,
) -> tuple[list[str], list[list[str]]] | None:
    item = root.find("./ReportSWInterface/section/item[@name='SW-to-HW Mapping']/table")
    if item is None:
        return None
    return _parse_report_table(item)


def _parse_report_table(table: ET.Element) -> tuple[list[str], list[list[str]]]:
    columns = [part.strip() for part in _text(table, "keys").split(",")]
    rows = []
    for column in table.findall("column"):
        row = [column.get("name", MISSING)]
        row.extend(part.strip() for part in (column.text or "").split(","))
        rows.append(_fit_row(row, len(columns)))
    return columns, rows


def _fit_row(row: list[str], size: int) -> list[str]:
    if len(row) < size:
        return row + [MISSING] * (size - len(row))
    if len(row) > size:
        return row[: size - 1] + [", ".join(row[size - 1 :])]
    return row


def _render_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    if rows:
        log_table(title, columns, rows)
