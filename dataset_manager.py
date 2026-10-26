#!/usr/bin/env python3
"""Dataset manager for the Morse-vibration project.

This program works on the files produced by serial_logger.py. It never edits
raw CSV or JSON files. Derived manifests, validation reports, event labels and
split assignments are written to separate folders.

Typical use:

    python dataset_manager.py --dataset MorseDataset all
    python dataset_manager.py --dataset MorseDataset split --seed 42

The `all` command builds a manifest, validates raw captures, generates event
labels, and rebuilds the manifest with validation results. Dataset splitting is
separate because a meaningful split requires at least three collection groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


REQUIRED_SAMPLE_COLUMNS = [
    "recording_id",
    "calibration_id",
    "sample_index",
    "timestamp_us",
    "accel_x_raw",
    "accel_y_raw",
    "accel_z_raw",
    "gyro_x_raw",
    "gyro_y_raw",
    "gyro_z_raw",
    "key_state",
    "motor_command",
    "system_state",
    "status_flags",
]

SENSOR_COLUMNS = [
    "accel_x_raw", "accel_y_raw", "accel_z_raw",
    "gyro_x_raw", "gyro_y_raw", "gyro_z_raw",
]

INTEGER_COLUMNS = set(REQUIRED_SAMPLE_COLUMNS)
EXPECTED_STATE = {"calibration": 1, "recording": 3}
ALLOWED_SPEEDS = {"slow", "normal", "fast"}
ALLOWED_STRENGTHS = {"low", "medium", "high"}
ALLOWED_NOISE = {"clean", "table_tap", "ambient_vibration", "movement", "other"}

MANIFEST_FIELDS = [
    "capture_uid", "capture_kind", "participant_id", "session_id",
    "target_text", "target_morse", "speed_condition", "drive_strength",
    "drive_command", "noise_condition", "calibration_ids",
    "recording_ids", "calibration_link_status", "linked_calibration_uid",
    "packet_count", "duration_us", "missing_samples", "crc_errors_total",
    "quality_status", "csv_path", "json_path",
]

VALIDATION_FIELDS = [
    "capture_uid", "capture_kind", "quality_status", "row_count",
    "duration_us", "effective_sample_rate_hz", "missing_samples",
    "duplicate_or_out_of_order", "nonmonotonic_timestamps", "parse_errors",
    "invalid_key_states", "unexpected_system_states", "motor_key_mismatches",
    "saturation_samples", "frozen_channels", "calibration_link_status",
    "errors", "warnings",
]

EVENT_FIELDS = [
    "capture_uid", "recording_id", "calibration_id", "event_index",
    "event_type", "expected_label", "start_us", "end_us", "duration_us",
    "start_sample_index", "end_sample_index", "alignment_status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value).strip("_") or "unnamed"


def semicolon(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in values)


def atomic_json_write(path: Path, value: Any) -> None:
    """Write derived JSON safely without touching any raw input file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return sorted(set(result))
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


@dataclass
class DatasetLayout:
    root: Path

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.calibrations = self.root / "calibrations"
        self.recordings = self.root / "recordings"
        self.processed = self.root / "processed"
        self.events = self.processed / "events"
        self.manifests = self.root / "manifests"
        self.splits = self.root / "splits"
        self.figures = self.root / "figures"
        self.logs = self.root / "logs"
        for directory in (
            self.calibrations, self.recordings, self.processed, self.events,
            self.manifests, self.splits, self.figures, self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path.resolve())


@dataclass
class Capture:
    uid: str
    kind: str
    csv_path: Path
    json_path: Path
    metadata: dict[str, Any]

    @property
    def session_id(self) -> str:
        return str(self.metadata.get("session_id", ""))

    @property
    def participant_id(self) -> str:
        return str(self.metadata.get("participant_id", ""))

    @property
    def calibration_ids(self) -> list[int]:
        return as_int_list(self.metadata.get("calibration_ids_seen"))

    @property
    def recording_ids(self) -> list[int]:
        return as_int_list(self.metadata.get("recording_ids_seen"))


def discover_captures(layout: DatasetLayout) -> tuple[list[Capture], list[str]]:
    captures: list[Capture] = []
    problems: list[str] = []

    for kind, directory in (("calibration", layout.calibrations), ("recording", layout.recordings)):
        for json_path in sorted(directory.glob("*.json")):
            csv_path = json_path.with_suffix(".csv")
            uid = f"{kind}:{json_path.stem}"
            if not csv_path.exists():
                problems.append(f"Missing CSV for {json_path}")
                continue
            try:
                metadata = load_json(json_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"Cannot read {json_path}: {exc}")
                continue
            declared_kind = metadata.get("capture_kind")
            if declared_kind not in (None, kind):
                problems.append(f"Capture kind mismatch in {json_path}: {declared_kind!r}")
            captures.append(Capture(uid, kind, csv_path, json_path, metadata))
    return captures, problems


def calibration_index(captures: Iterable[Capture]) -> dict[tuple[str, int], list[Capture]]:
    index: dict[tuple[str, int], list[Capture]] = {}
    for capture in captures:
        if capture.kind != "calibration":
            continue
        for calibration_id in capture.calibration_ids:
            index.setdefault((capture.session_id, calibration_id), []).append(capture)
    return index


def calibration_link(capture: Capture, index: dict[tuple[str, int], list[Capture]]) -> tuple[str, str]:
    if capture.kind == "calibration":
        return "self", capture.uid
    ids = capture.calibration_ids
    if len(ids) != 1:
        return "ambiguous_ids" if ids else "missing_id", ""
    matches = index.get((capture.session_id, ids[0]), [])
    if len(matches) == 1:
        return "linked", matches[0].uid
    if not matches:
        return "missing_calibration", ""
    return "duplicate_calibration", ""


def read_validation_map(layout: DatasetLayout) -> dict[str, dict[str, Any]]:
    path = layout.manifests / "validation_report.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        results = value.get("captures", []) if isinstance(value, dict) else []
        return {str(item["capture_uid"]): item for item in results if isinstance(item, dict) and "capture_uid" in item}
    except (OSError, json.JSONDecodeError):
        return {}


def manifest_row(
    layout: DatasetLayout,
    capture: Capture,
    cal_index: dict[tuple[str, int], list[Capture]],
    validation: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    link_status, linked_uid = calibration_link(capture, cal_index)
    checked = validation.get(capture.uid, {})
    return {
        "capture_uid": capture.uid,
        "capture_kind": capture.kind,
        "participant_id": capture.participant_id,
        "session_id": capture.session_id,
        "target_text": capture.metadata.get("target_text", ""),
        "target_morse": capture.metadata.get("target_morse", ""),
        "speed_condition": capture.metadata.get("speed_condition", ""),
        "drive_strength": capture.metadata.get("drive_strength", ""),
        "drive_command": capture.metadata.get("drive_command", ""),
        "noise_condition": capture.metadata.get("noise_condition", ""),
        "calibration_ids": semicolon(capture.calibration_ids),
        "recording_ids": semicolon(capture.recording_ids),
        "calibration_link_status": link_status,
        "linked_calibration_uid": linked_uid,
        "packet_count": capture.metadata.get("packet_count", ""),
        "duration_us": checked.get("duration_us", capture.metadata.get("duration_us", "")),
        "missing_samples": checked.get("missing_samples", capture.metadata.get("missing_samples", "")),
        "crc_errors_total": capture.metadata.get("parser_crc_errors_total", ""),
        "quality_status": checked.get("quality_status", "not_validated"),
        "csv_path": layout.relative(capture.csv_path),
        "json_path": layout.relative(capture.json_path),
    }


def build_manifest(layout: DatasetLayout) -> Path:
    captures, discovery_problems = discover_captures(layout)
    cal_index = calibration_index(captures)
    validation = read_validation_map(layout)
    rows = [manifest_row(layout, capture, cal_index, validation) for capture in captures]
    path = layout.manifests / "master_manifest.csv"
    write_csv(path, MANIFEST_FIELDS, rows)
    summary = {
        "generated_utc": utc_now(),
        "dataset_root": str(layout.root),
        "capture_count": len(captures),
        "calibration_count": sum(c.kind == "calibration" for c in captures),
        "recording_count": sum(c.kind == "recording" for c in captures),
        "discovery_problems": discovery_problems,
        "manifest_file": layout.relative(path),
    }
    atomic_json_write(layout.manifests / "manifest_summary.json", summary)
    print(f"Manifest: {path} ({len(rows)} captures)")
    for problem in discovery_problems:
        print(f"WARNING: {problem}")
    return path


@dataclass
class ValidationResult:
    capture_uid: str
    capture_kind: str
    quality_status: str = "pass"
    row_count: int = 0
    duration_us: int = 0
    effective_sample_rate_hz: float = 0.0
    missing_samples: int = 0
    duplicate_or_out_of_order: int = 0
    nonmonotonic_timestamps: int = 0
    parse_errors: int = 0
    invalid_key_states: int = 0
    unexpected_system_states: int = 0
    motor_key_mismatches: int = 0
    saturation_samples: int = 0
    frozen_channels: list[str] = field(default_factory=list)
    calibration_link_status: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def finish(self) -> None:
        if self.errors:
            self.quality_status = "fail"
        elif self.warnings:
            self.quality_status = "warning"
        else:
            self.quality_status = "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "capture_uid": self.capture_uid,
            "capture_kind": self.capture_kind,
            "quality_status": self.quality_status,
            "row_count": self.row_count,
            "duration_us": self.duration_us,
            "effective_sample_rate_hz": round(self.effective_sample_rate_hz, 3),
            "missing_samples": self.missing_samples,
            "duplicate_or_out_of_order": self.duplicate_or_out_of_order,
            "nonmonotonic_timestamps": self.nonmonotonic_timestamps,
            "parse_errors": self.parse_errors,
            "invalid_key_states": self.invalid_key_states,
            "unexpected_system_states": self.unexpected_system_states,
            "motor_key_mismatches": self.motor_key_mismatches,
            "saturation_samples": self.saturation_samples,
            "frozen_channels": self.frozen_channels,
            "calibration_link_status": self.calibration_link_status,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def csv_dict(self) -> dict[str, Any]:
        value = self.as_dict()
        value["frozen_channels"] = semicolon(value["frozen_channels"])
        value["errors"] = " | ".join(value["errors"])
        value["warnings"] = " | ".join(value["warnings"])
        return value


def validate_metadata(capture: Capture, result: ValidationResult) -> None:
    metadata = capture.metadata
    for field_name in ("participant_id", "session_id"):
        if not str(metadata.get(field_name, "")).strip():
            result.errors.append(f"missing metadata: {field_name}")

    if capture.kind == "recording":
        for field_name in ("target_text", "target_morse"):
            if not str(metadata.get(field_name, "")).strip():
                result.errors.append(f"missing metadata: {field_name}")
        speed = str(metadata.get("speed_condition", ""))
        strength = str(metadata.get("drive_strength", ""))
        noise = str(metadata.get("noise_condition", ""))
        if speed not in ALLOWED_SPEEDS:
            result.warnings.append(f"unexpected speed_condition: {speed!r}")
        if strength not in ALLOWED_STRENGTHS:
            result.warnings.append(f"unexpected drive_strength: {strength!r}")
        if noise not in ALLOWED_NOISE:
            result.warnings.append(f"unexpected noise_condition: {noise!r}")


def validate_capture(
    capture: Capture,
    cal_index: dict[tuple[str, int], list[Capture]],
    nominal_rate_hz: float,
    rate_tolerance: float,
) -> ValidationResult:
    result = ValidationResult(capture.uid, capture.kind)
    validate_metadata(capture, result)
    link_status, _ = calibration_link(capture, cal_index)
    result.calibration_link_status = link_status
    if capture.kind == "recording" and link_status != "linked":
        result.errors.append(f"calibration link is {link_status}")

    sensor_min = {name: math.inf for name in SENSOR_COLUMNS}
    sensor_max = {name: -math.inf for name in SENSOR_COLUMNS}
    first_timestamp: Optional[int] = None
    previous_timestamp: Optional[int] = None
    previous_index: Optional[int] = None
    expected_state = EXPECTED_STATE[capture.kind]

    try:
        with capture.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            missing_columns = [name for name in REQUIRED_SAMPLE_COLUMNS if name not in fields]
            if missing_columns:
                result.errors.append(f"missing columns: {', '.join(missing_columns)}")
                result.finish()
                return result

            for raw in reader:
                try:
                    row = {name: int(raw[name]) for name in INTEGER_COLUMNS}
                except (KeyError, TypeError, ValueError):
                    result.parse_errors += 1
                    continue

                result.row_count += 1
                index = row["sample_index"]
                timestamp = row["timestamp_us"]
                if first_timestamp is None:
                    first_timestamp = timestamp

                if previous_index is not None:
                    difference = index - previous_index
                    if difference > 1:
                        result.missing_samples += difference - 1
                    elif difference <= 0:
                        result.duplicate_or_out_of_order += 1
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    result.nonmonotonic_timestamps += 1
                previous_index = index
                previous_timestamp = timestamp

                if row["key_state"] not in (0, 1):
                    result.invalid_key_states += 1
                if row["system_state"] != expected_state:
                    result.unexpected_system_states += 1
                if row["key_state"] == 0 and row["motor_command"] != 0:
                    result.motor_key_mismatches += 1

                saturated = False
                for name in SENSOR_COLUMNS:
                    value = row[name]
                    sensor_min[name] = min(sensor_min[name], value)
                    sensor_max[name] = max(sensor_max[name], value)
                    saturated = saturated or abs(value) >= 32760
                if saturated:
                    result.saturation_samples += 1
    except OSError as exc:
        result.errors.append(f"cannot read CSV: {exc}")
        result.finish()
        return result

    if result.row_count == 0:
        result.errors.append("capture contains no valid rows")
    if result.parse_errors:
        result.errors.append(f"{result.parse_errors} rows could not be parsed")
    if first_timestamp is not None and previous_timestamp is not None:
        result.duration_us = max(0, previous_timestamp - first_timestamp)
        if result.duration_us > 0 and result.row_count > 1:
            result.effective_sample_rate_hz = (result.row_count - 1) * 1_000_000.0 / result.duration_us
            relative_error = abs(result.effective_sample_rate_hz - nominal_rate_hz) / nominal_rate_hz
            if relative_error > rate_tolerance:
                result.warnings.append(
                    f"effective sample rate {result.effective_sample_rate_hz:.1f} Hz "
                    f"differs from nominal {nominal_rate_hz:.1f} Hz"
                )

    result.frozen_channels = [
        name for name in SENSOR_COLUMNS
        if result.row_count > 1 and sensor_min[name] == sensor_max[name]
    ]
    if result.frozen_channels:
        result.warnings.append(f"frozen channels: {', '.join(result.frozen_channels)}")
    if result.missing_samples:
        fraction = result.missing_samples / max(1, result.row_count + result.missing_samples)
        message = f"{result.missing_samples} missing samples ({fraction:.3%})"
        if fraction > 0.01:
            result.errors.append(message)
        else:
            result.warnings.append(message)
    if result.duplicate_or_out_of_order:
        result.errors.append(f"{result.duplicate_or_out_of_order} duplicate/out-of-order indices")
    if result.nonmonotonic_timestamps:
        result.errors.append(f"{result.nonmonotonic_timestamps} nonmonotonic timestamps")
    if result.invalid_key_states:
        result.errors.append(f"{result.invalid_key_states} invalid key states")
    if result.unexpected_system_states:
        result.errors.append(f"{result.unexpected_system_states} unexpected system states")
    if result.motor_key_mismatches:
        result.warnings.append(f"{result.motor_key_mismatches} samples have motor on while key is released")
    if result.saturation_samples:
        result.warnings.append(f"{result.saturation_samples} samples contain saturated sensor values")

    logger_crc = capture.metadata.get("parser_crc_errors_total", 0)
    try:
        if int(logger_crc) > 0:
            result.warnings.append(f"logger reported {logger_crc} cumulative CRC errors")
    except (TypeError, ValueError):
        result.warnings.append("invalid parser_crc_errors_total metadata")

    result.finish()
    return result


def validate_dataset(layout: DatasetLayout, nominal_rate_hz: float, rate_tolerance: float) -> Path:
    captures, discovery_problems = discover_captures(layout)
    cal_index = calibration_index(captures)
    results = [
        validate_capture(capture, cal_index, nominal_rate_hz, rate_tolerance)
        for capture in captures
    ]
    report = {
        "generated_utc": utc_now(),
        "nominal_sample_rate_hz": nominal_rate_hz,
        "sample_rate_tolerance_fraction": rate_tolerance,
        "discovery_problems": discovery_problems,
        "summary": {
            "captures": len(results),
            "pass": sum(r.quality_status == "pass" for r in results),
            "warning": sum(r.quality_status == "warning" for r in results),
            "fail": sum(r.quality_status == "fail" for r in results),
        },
        "captures": [result.as_dict() for result in results],
    }
    json_path = layout.manifests / "validation_report.json"
    csv_path = layout.manifests / "validation_summary.csv"
    atomic_json_write(json_path, report)
    write_csv(csv_path, VALIDATION_FIELDS, (result.csv_dict() for result in results))
    print(
        f"Validation: {report['summary']['pass']} pass, "
        f"{report['summary']['warning']} warning, {report['summary']['fail']} fail"
    )
    print(f"Validation report: {json_path}")
    return json_path


@dataclass
class StateRun:
    state: int
    start_us: int
    end_us: int
    start_index: int
    end_index: int
    recording_id: int
    calibration_id: int

    @property
    def duration_us(self) -> int:
        return max(0, self.end_us - self.start_us)


def load_key_runs(csv_path: Path) -> list[StateRun]:
    runs: list[StateRun] = []
    current: Optional[StateRun] = None
    intervals: list[int] = []
    previous_timestamp: Optional[int] = None

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                timestamp = int(raw["timestamp_us"])
                index = int(raw["sample_index"])
                state = int(raw["key_state"])
                recording_id = int(raw["recording_id"])
                calibration_id = int(raw["calibration_id"])
            except (KeyError, TypeError, ValueError):
                continue

            if previous_timestamp is not None and timestamp > previous_timestamp:
                intervals.append(timestamp - previous_timestamp)
            previous_timestamp = timestamp

            if current is None:
                current = StateRun(state, timestamp, timestamp, index, index, recording_id, calibration_id)
            elif state == current.state:
                current.end_us = timestamp
                current.end_index = index
            else:
                current.end_us = timestamp
                runs.append(current)
                current = StateRun(state, timestamp, timestamp, index, index, recording_id, calibration_id)

    if current is not None:
        median_interval = int(statistics.median(intervals)) if intervals else 0
        current.end_us += median_interval
        runs.append(current)
    return runs


def expected_event_labels(target_morse: str) -> tuple[list[str], list[str]]:
    """Return pulse labels and labels for the gaps between consecutive pulses."""
    words = [word.strip() for word in target_morse.strip().split("/") if word.strip()]
    pulses: list[str] = []
    gaps: list[str] = []
    for word_index, word in enumerate(words):
        characters = [item for item in word.split() if item]
        for character_index, code in enumerate(characters):
            symbols = [symbol for symbol in code if symbol in ".-"]
            for symbol_index, symbol in enumerate(symbols):
                pulses.append("dot" if symbol == "." else "dash")
                is_last_symbol = symbol_index == len(symbols) - 1
                is_last_character = character_index == len(characters) - 1
                is_last_word = word_index == len(words) - 1
                if not is_last_symbol:
                    gaps.append("element_gap")
                elif not is_last_character:
                    gaps.append("character_gap")
                elif not is_last_word:
                    gaps.append("word_gap")
    return pulses, gaps


def generate_events_for_capture(layout: DatasetLayout, capture: Capture) -> tuple[Path, dict[str, Any]]:
    runs = load_key_runs(capture.csv_path)
    pulse_positions = [index for index, run in enumerate(runs) if run.state == 1]
    target_pulses, target_gaps = expected_event_labels(str(capture.metadata.get("target_morse", "")))

    if not pulse_positions:
        selected_runs: list[StateRun] = []
    else:
        selected_runs = runs[pulse_positions[0]:pulse_positions[-1] + 1]

    observed_pulses = sum(run.state == 1 for run in selected_runs)
    observed_gaps = sum(run.state == 0 for run in selected_runs)
    aligned = observed_pulses == len(target_pulses) and observed_gaps == len(target_gaps)
    alignment_status = "exact" if aligned else "mismatch"

    rows: list[dict[str, Any]] = []
    pulse_index = 0
    gap_index = 0
    for event_index, run in enumerate(selected_runs):
        if run.state == 1:
            label = target_pulses[pulse_index] if aligned else "unmatched_pulse"
            pulse_index += 1
            event_type = "pulse"
        else:
            label = target_gaps[gap_index] if aligned else "unmatched_gap"
            gap_index += 1
            event_type = "gap"
        rows.append({
            "capture_uid": capture.uid,
            "recording_id": run.recording_id,
            "calibration_id": run.calibration_id,
            "event_index": event_index,
            "event_type": event_type,
            "expected_label": label,
            "start_us": run.start_us,
            "end_us": run.end_us,
            "duration_us": run.duration_us,
            "start_sample_index": run.start_index,
            "end_sample_index": run.end_index,
            "alignment_status": alignment_status,
        })

    output = layout.events / f"{safe_name(capture.json_path.stem)}_events.csv"
    write_csv(output, EVENT_FIELDS, rows)
    summary = {
        "capture_uid": capture.uid,
        "event_file": layout.relative(output),
        "alignment_status": alignment_status,
        "observed_pulses": observed_pulses,
        "expected_pulses": len(target_pulses),
        "observed_gaps": observed_gaps,
        "expected_gaps": len(target_gaps),
    }
    return output, summary


def generate_all_events(layout: DatasetLayout) -> Path:
    captures, _ = discover_captures(layout)
    recordings = [capture for capture in captures if capture.kind == "recording"]
    summaries: list[dict[str, Any]] = []
    for capture in recordings:
        _, summary = generate_events_for_capture(layout, capture)
        summaries.append(summary)
    path = layout.manifests / "event_generation_report.json"
    atomic_json_write(path, {"generated_utc": utc_now(), "recordings": summaries})
    exact = sum(item["alignment_status"] == "exact" for item in summaries)
    print(f"Events: {exact}/{len(summaries)} recordings aligned exactly")
    print(f"Event report: {path}")
    return path


def stable_capture_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def allocate_groups(groups: list[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, str]:
    if len(groups) < 3:
        raise ValueError("At least three sessions/participants are required for train/validation/test splitting")
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    validation_count = max(1, round(count * val_ratio))
    test_count = max(1, round(count * (1.0 - train_ratio - val_ratio)))
    if validation_count + test_count >= count:
        validation_count = 1
        test_count = 1
    train_count = count - validation_count - test_count
    assignment: dict[str, str] = {}
    for group in shuffled[:train_count]:
        assignment[group] = "train"
    for group in shuffled[train_count:train_count + validation_count]:
        assignment[group] = "validation"
    for group in shuffled[train_count + validation_count:]:
        assignment[group] = "test"
    return assignment


def create_split(
    layout: DatasetLayout,
    seed: int,
    group_by: str,
    train_ratio: float,
    val_ratio: float,
) -> Path:
    captures, _ = discover_captures(layout)
    validation = read_validation_map(layout)
    recordings = [
        capture for capture in captures
        if capture.kind == "recording"
        and validation.get(capture.uid, {}).get("quality_status", "not_validated") != "fail"
    ]
    if not recordings:
        raise ValueError("No usable recordings found. Run validation after collecting data")
    groups = sorted({str(getattr(capture, group_by)) for capture in recordings if str(getattr(capture, group_by))})
    assignment = allocate_groups(groups, seed, train_ratio, val_ratio)
    rows = []
    for capture in recordings:
        group = str(getattr(capture, group_by))
        rows.append({
            "capture_uid": capture.uid,
            "participant_id": capture.participant_id,
            "session_id": capture.session_id,
            "group_by": group_by,
            "group_value": group,
            "split": assignment[group],
            "stable_capture_hash": stable_capture_hash(capture.uid),
        })
    fields = ["capture_uid", "participant_id", "session_id", "group_by", "group_value", "split", "stable_capture_hash"]
    path = layout.splits / f"{group_by}_split_seed{seed}.csv"
    write_csv(path, fields, rows)
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    summary = {
        "generated_utc": utc_now(),
        "seed": seed,
        "group_by": group_by,
        "group_assignments": assignment,
        "recording_counts": counts,
        "split_file": layout.relative(path),
    }
    atomic_json_write(path.with_suffix(".json"), summary)
    print(f"Split: {counts}; file: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage and validate the Morse-vibration dataset")
    parser.add_argument("--dataset", type=Path, default=Path("MorseDataset"), help="Dataset root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("manifest", help="Build the master capture manifest")

    validate = subparsers.add_parser("validate", help="Validate all raw CSV/JSON capture pairs")
    validate.add_argument("--nominal-rate", type=float, default=833.0)
    validate.add_argument("--rate-tolerance", type=float, default=0.05, help="Allowed fractional rate error")

    subparsers.add_parser("events", help="Generate pulse and gap labels from key_state")

    split = subparsers.add_parser("split", help="Create leakage-safe group-level splits")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--group-by", choices=("session_id", "participant_id"), default="session_id")
    split.add_argument("--train-ratio", type=float, default=0.70)
    split.add_argument("--validation-ratio", type=float, default=0.15)

    all_parser = subparsers.add_parser("all", help="Run manifest, validation and event generation")
    all_parser.add_argument("--nominal-rate", type=float, default=833.0)
    all_parser.add_argument("--rate-tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = DatasetLayout(args.dataset)
    try:
        if args.command == "manifest":
            build_manifest(layout)
        elif args.command == "validate":
            validate_dataset(layout, args.nominal_rate, args.rate_tolerance)
            build_manifest(layout)
        elif args.command == "events":
            generate_all_events(layout)
        elif args.command == "split":
            if args.train_ratio <= 0 or args.validation_ratio <= 0:
                raise ValueError("Split ratios must be positive")
            if args.train_ratio + args.validation_ratio >= 1:
                raise ValueError("Train ratio plus validation ratio must be less than 1")
            create_split(
                layout, args.seed, args.group_by,
                args.train_ratio, args.validation_ratio,
            )
        elif args.command == "all":
            build_manifest(layout)
            validate_dataset(layout, args.nominal_rate, args.rate_tolerance)
            generate_all_events(layout)
            build_manifest(layout)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
