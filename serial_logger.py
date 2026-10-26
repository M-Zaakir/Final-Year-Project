#!/usr/bin/env python3
"""PC-side serial logger for the ESP32-S3 Morse-vibration rig.

Expected firmware packet:
    magic/version/type/recording_id/calibration_id/sample_index/timestamp_us/
    accel_xyz/gyro_xyz/key_state/motor_command/system_state/status_flags/crc16

The firmware sends 42-byte little-endian packets at up to 1660 samples/s.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("pyserial is not installed. Run: python -m pip install pyserial")
    raise SystemExit(1)


MAGIC = 0xA55A
MAGIC_BYTES = b"\x5A\xA5"
PACKET_VERSION = 3
PACKET_TYPE_SAMPLE = 1
PACKET_STRUCT = struct.Struct("<HBBIIIQhhhhhhBBBBH")
PACKET_SIZE = PACKET_STRUCT.size
CRC_DATA_SIZE = PACKET_SIZE - 2

STATE_UNCALIBRATED = 0
STATE_CALIBRATING = 1
STATE_READY = 2
STATE_RECORDING = 3

STRENGTHS = {
    "1": {"label": "low", "command": 64},
    "2": {"label": "medium", "command": 128},
    "3": {"label": "high", "command": 192},
}

CSV_FIELDS = [
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

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".",
    "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
    "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---",
    "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--",
    "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....",
    "7": "--...", "8": "---..", "9": "----.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "unnamed"


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def prompt_choice(label: str, choices: tuple[str, ...], default: str) -> str:
    while True:
        value = prompt(f"{label} ({'/'.join(choices)})", default).lower()
        if value in choices:
            return value
        print(f"Choose one of: {', '.join(choices)}")


def prompt_int(label: str, default: int, allow_blank: bool = False) -> Optional[int]:
    while True:
        shown_default = "" if allow_blank else str(default)
        value = prompt(label, shown_default)
        if allow_blank and value == "":
            return None
        try:
            return int(value)
        except ValueError:
            print("Enter a whole number.")


def morse_encode(text: str) -> str:
    words = []
    for word in text.upper().split():
        symbols = [MORSE[ch] for ch in word if ch in MORSE]
        words.append(" ".join(symbols))
    return " / ".join(words)


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class SamplePacket:
    magic: int
    version: int
    packet_type: int
    recording_id: int
    calibration_id: int
    sample_index: int
    timestamp_us: int
    accel_x_raw: int
    accel_y_raw: int
    accel_z_raw: int
    gyro_x_raw: int
    gyro_y_raw: int
    gyro_z_raw: int
    key_state: int
    motor_command: int
    system_state: int
    status_flags: int
    crc16: int

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SamplePacket":
        return cls(*PACKET_STRUCT.unpack(raw))

    def csv_row(self) -> dict[str, int]:
        values = asdict(self)
        return {field: values[field] for field in CSV_FIELDS}


class PacketParser:
    """Resynchronising parser that tolerates startup text and corrupt bytes."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.valid_packets = 0
        self.crc_errors = 0
        self.version_errors = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> list[SamplePacket]:
        self.buffer.extend(data)
        packets: list[SamplePacket] = []

        while True:
            position = self.buffer.find(MAGIC_BYTES)
            if position < 0:
                # Retain a possible first magic byte at the end.
                keep = 1 if self.buffer.endswith(MAGIC_BYTES[:1]) else 0
                discard = len(self.buffer) - keep
                if discard > 0:
                    del self.buffer[:discard]
                    self.discarded_bytes += discard
                break

            if position > 0:
                del self.buffer[:position]
                self.discarded_bytes += position

            if len(self.buffer) < PACKET_SIZE:
                break

            raw = bytes(self.buffer[:PACKET_SIZE])
            received_crc = int.from_bytes(raw[-2:], "little")
            calculated_crc = crc16_ccitt_false(raw[:CRC_DATA_SIZE])
            if received_crc != calculated_crc:
                self.crc_errors += 1
                del self.buffer[0]
                continue

            packet = SamplePacket.from_bytes(raw)
            if packet.magic != MAGIC or packet.packet_type != PACKET_TYPE_SAMPLE:
                del self.buffer[0]
                self.discarded_bytes += 1
                continue
            if packet.version != PACKET_VERSION:
                self.version_errors += 1
                del self.buffer[:PACKET_SIZE]
                continue

            packets.append(packet)
            self.valid_packets += 1
            del self.buffer[:PACKET_SIZE]

        return packets


class DatasetLayout:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.calibrations = self.root / "calibrations"
        self.recordings = self.root / "recordings"
        self.processed = self.root / "processed"
        self.manifests = self.root / "manifests"
        self.splits = self.root / "splits"
        self.figures = self.root / "figures"
        self.logs = self.root / "logs"
        for directory in (
            self.calibrations, self.recordings, self.processed,
            self.manifests, self.splits, self.figures, self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class CaptureWriter:
    def __init__(
        self,
        directory: Path,
        base_name: str,
        capture_kind: str,
        metadata: dict[str, Any],
    ) -> None:
        self.capture_kind = capture_kind
        self.csv_path = directory / f"{base_name}.csv"
        self.json_path = directory / f"{base_name}.json"
        self.file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.metadata = metadata
        self.started_utc = utc_now()
        self.first_packet_monotonic: Optional[float] = None
        self.last_packet_monotonic: Optional[float] = None
        self.first_sample_index: Optional[int] = None
        self.last_sample_index: Optional[int] = None
        self.packet_count = 0
        self.missing_samples = 0
        self.duplicate_or_out_of_order = 0
        self.unexpected_state_packets = 0
        self.calibration_ids: set[int] = set()
        self.recording_ids: set[int] = set()
        self.motor_commands: set[int] = set()
        self.key_pressed_samples = 0
        self._closed = False

    def accepts(self, packet: SamplePacket) -> bool:
        expected = STATE_CALIBRATING if self.capture_kind == "calibration" else STATE_RECORDING
        return packet.system_state == expected

    def write(self, packet: SamplePacket) -> None:
        if self._closed:
            return
        if not self.accepts(packet):
            self.unexpected_state_packets += 1
            return

        now = time.monotonic()
        if self.first_packet_monotonic is None:
            self.first_packet_monotonic = now
            self.first_sample_index = packet.sample_index
        self.last_packet_monotonic = now

        if self.last_sample_index is not None:
            difference = packet.sample_index - self.last_sample_index
            if difference > 1:
                self.missing_samples += difference - 1
            elif difference <= 0:
                self.duplicate_or_out_of_order += 1
        self.last_sample_index = packet.sample_index

        self.writer.writerow(packet.csv_row())
        self.packet_count += 1
        self.calibration_ids.add(packet.calibration_id)
        self.recording_ids.add(packet.recording_id)
        self.motor_commands.add(packet.motor_command)
        self.key_pressed_samples += packet.key_state

        if self.packet_count % 500 == 0:
            self.file.flush()

    def close(self, parser: PacketParser, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        self.file.flush()
        self.file.close()

        duration_us = None
        if self.packet_count > 0:
            # CSV timestamps are relative to the ESP32 capture start.
            duration_us = self.metadata.get("last_timestamp_us")

        summary = {
            **self.metadata,
            "capture_kind": self.capture_kind,
            "started_utc": self.started_utc,
            "finished_utc": utc_now(),
            "close_reason": reason,
            "csv_file": self.csv_path.name,
            "packet_version": PACKET_VERSION,
            "packet_size_bytes": PACKET_SIZE,
            "packet_count": self.packet_count,
            "first_sample_index": self.first_sample_index,
            "last_sample_index": self.last_sample_index,
            "missing_samples": self.missing_samples,
            "duplicate_or_out_of_order": self.duplicate_or_out_of_order,
            "unexpected_state_packets": self.unexpected_state_packets,
            "calibration_ids_seen": sorted(self.calibration_ids),
            "recording_ids_seen": sorted(self.recording_ids),
            "motor_commands_seen": sorted(self.motor_commands),
            "key_pressed_samples": self.key_pressed_samples,
            "duration_us": duration_us,
            "parser_crc_errors_total": parser.crc_errors,
            "parser_version_errors_total": parser.version_errors,
            "parser_discarded_bytes_total": parser.discarded_bytes,
        }
        self.json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


class SerialLogger:
    def __init__(
        self,
        port: str,
        baud: int,
        layout: DatasetLayout,
        session_metadata: dict[str, Any],
    ) -> None:
        self.port = port
        self.baud = baud
        self.layout = layout
        self.session_metadata = session_metadata
        self.serial = serial.Serial(port=port, baudrate=baud, timeout=0.05, write_timeout=1.0)
        self.parser = PacketParser()
        self.stop_event = threading.Event()
        self.reader_exception: queue.Queue[BaseException] = queue.Queue()
        self.writer_lock = threading.Lock()
        self.active_writer: Optional[CaptureWriter] = None
        self.current_strength_key = "2"
        self.reader = threading.Thread(target=self._reader_loop, name="serial-reader", daemon=True)
        self.reader.start()

    def _reader_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                data = self.serial.read(4096)
                if not data:
                    continue
                for packet in self.parser.feed(data):
                    with self.writer_lock:
                        if self.active_writer is not None:
                            self.active_writer.metadata["last_timestamp_us"] = packet.timestamp_us
                            self.active_writer.write(packet)
        except BaseException as exc:
            self.reader_exception.put(exc)
            self.stop_event.set()

    def check_reader(self) -> None:
        if not self.reader_exception.empty():
            raise RuntimeError("Serial reader stopped") from self.reader_exception.get()

    def send(self, command: str) -> None:
        self.check_reader()
        self.serial.write(command.encode("ascii"))
        self.serial.flush()

    def enter_binary_mode(self) -> None:
        self.serial.reset_input_buffer()
        self.send("b")
        time.sleep(0.3)
        # Remove the firmware's final human-readable line before packet capture.
        self.serial.reset_input_buffer()
        self.parser.buffer.clear()

    def _new_writer(self, kind: str, metadata: dict[str, Any]) -> CaptureWriter:
        if self.active_writer is not None:
            raise RuntimeError("A capture is already active")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = safe_name(str(self.session_metadata["session_id"]))
        prefix = "calibration" if kind == "calibration" else "recording"
        base_name = f"{session_id}_{prefix}_{stamp}"
        directory = self.layout.calibrations if kind == "calibration" else self.layout.recordings
        writer = CaptureWriter(directory, base_name, kind, metadata)
        self.active_writer = writer
        return writer

    def _close_writer(self, reason: str) -> Optional[CaptureWriter]:
        with self.writer_lock:
            writer = self.active_writer
            if writer is not None:
                writer.close(self.parser, reason)
                self.active_writer = None
            return writer

    def calibrate(self) -> None:
        if self.active_writer is not None:
            print("A capture is already active.")
            return
        notes = prompt("Calibration/environment notes", "rig undisturbed")
        metadata = {
            **self.session_metadata,
            "environment_notes": notes,
            "requested_duration_s": 5,
            "firmware_command": "c",
        }
        writer = self._new_writer("calibration", metadata)
        print("Keep the rig completely still. Starting five-second calibration...")
        self.send("c")

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            self.check_reader()
            time.sleep(0.1)
            if writer.packet_count > 0 and writer.last_packet_monotonic is not None:
                if time.monotonic() - writer.last_packet_monotonic > 0.8:
                    break

        finished = self._close_writer("calibration_complete_or_stream_stopped")
        if finished is None or finished.packet_count == 0:
            print("WARNING: no calibration packets were received. Check firmware state and wiring.")
        else:
            print(f"Calibration saved: {finished.csv_path}")
            print(f"Packets: {finished.packet_count}; missing samples: {finished.missing_samples}")

    def start_recording(self) -> None:
        if self.active_writer is not None:
            print("A capture is already active. Use 'stop' first.")
            return

        target_text = prompt("Target text").upper()
        if not target_text:
            print("A target text label is required.")
            return
        target_morse = morse_encode(target_text)
        print(f"Target Morse: {target_morse}")
        speed = prompt_choice("Speed condition", ("slow", "normal", "fast"), "normal")
        intended_dot_ms = prompt_int("Intended dot duration in ms (optional)", 0, allow_blank=True)
        noise = prompt_choice(
            "Noise condition",
            ("clean", "table_tap", "ambient_vibration", "movement", "other"),
            "clean",
        )
        noise_details = prompt("Noise details", "none" if noise == "clean" else "")
        notes = prompt("Recording notes", "")
        strength = STRENGTHS[self.current_strength_key]

        metadata = {
            **self.session_metadata,
            "target_text": target_text,
            "target_morse": target_morse,
            "speed_condition": speed,
            "intended_dot_duration_ms": intended_dot_ms,
            "drive_strength": strength["label"],
            "drive_command": strength["command"],
            "noise_condition": noise,
            "noise_details": noise_details,
            "recording_notes": notes,
            "firmware_command": "r",
        }
        writer = self._new_writer("recording", metadata)
        self.send("r")
        print(f"Recording started: {writer.csv_path.name}")
        print("Transmit the message, then type: stop")

    def stop_recording(self) -> None:
        if self.active_writer is None or self.active_writer.capture_kind != "recording":
            print("No recording is active.")
            return
        self.send("r")
        time.sleep(0.5)
        writer = self._close_writer("operator_stop")
        if writer is not None:
            print(f"Recording saved: {writer.csv_path}")
            print(f"Packets: {writer.packet_count}; missing samples: {writer.missing_samples}")
            if writer.packet_count == 0:
                print("WARNING: no packets received. Calibration may be missing or recording was refused.")

    def set_strength(self, key: str) -> None:
        if key not in STRENGTHS:
            print("Strength must be 1, 2 or 3.")
            return
        if self.active_writer is not None:
            print("Do not change strength during a capture. Stop first.")
            return
        self.send(key)
        self.current_strength_key = key
        strength = STRENGTHS[key]
        print(f"Strength set to {strength['label']} ({strength['command']}/255).")

    def print_status(self) -> None:
        with self.writer_lock:
            writer = self.active_writer
            print(f"Port: {self.port} at {self.baud} baud")
            print(f"Valid packets received: {self.parser.valid_packets}")
            print(f"CRC errors: {self.parser.crc_errors}")
            print(f"Version errors: {self.parser.version_errors}")
            print(f"Discarded non-packet bytes: {self.parser.discarded_bytes}")
            if writer:
                print(f"Active: {writer.capture_kind}; packets={writer.packet_count}; missing={writer.missing_samples}")
            else:
                print("Active capture: none")

    def close(self) -> None:
        if self.active_writer is not None:
            if self.active_writer.capture_kind == "recording":
                try:
                    self.send("r")
                    time.sleep(0.3)
                except Exception:
                    pass
            self._close_writer("logger_shutdown")
        self.stop_event.set()
        self.reader.join(timeout=1.0)
        try:
            self.serial.close()
        except Exception:
            pass


def choose_port(requested: Optional[str]) -> str:
    if requested:
        return requested
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found. Connect the ESP32 or pass --port COMx.")
    print("Available serial ports:")
    for index, item in enumerate(ports, start=1):
        print(f"  {index}. {item.device} - {item.description}")
    while True:
        selected = prompt("Select port number", "1")
        try:
            return ports[int(selected) - 1].device
        except (ValueError, IndexError):
            print("Enter one of the listed numbers.")


def collect_session_metadata(participant_arg: Optional[str], session_arg: Optional[str]) -> dict[str, Any]:
    default_session = datetime.now().strftime("S%Y%m%d_%H%M")
    participant = participant_arg or prompt("Participant ID", "P01")
    session_id = session_arg or prompt("Session ID", default_session)
    return {
        "participant_id": participant,
        "session_id": session_id,
        "session_started_utc": utc_now(),
        "surface_material": prompt("Transmission medium", "stainless_steel_ruler"),
        "surface_dimensions_mm": prompt("Medium dimensions (LxWxT mm)", "600x30x1.2"),
        "clamped_length_mm": prompt_int("Clamped length in mm", 250),
        "free_length_mm": prompt_int("Free length in mm", 350),
        "actuator_position_from_table_edge_mm": prompt_int("LRA position beyond table edge in mm", 30),
        "sensor_position_from_table_edge_mm": prompt_int("IMU position beyond table edge in mm", 300),
        "sensor_orientation": prompt("Sensor orientation/axis notes", "record from marked reference orientation"),
        "mounting_method": prompt("Temporary/final mounting method", "temporary rigid mount"),
        "location": prompt("Collection location", "UCT laboratory"),
        "operator_notes": prompt("Session notes", ""),
        "nominal_sample_rate_hz": 833,
        "firmware_packet_version": PACKET_VERSION,
    }


def write_session_metadata(layout: DatasetLayout, metadata: dict[str, Any]) -> Path:
    path = layout.manifests / f"session_{safe_name(str(metadata['session_id']))}.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def print_commands() -> None:
    print(
        "\nCommands:\n"
        "  calibrate       collect the five-second background calibration\n"
        "  record          enter labels and start a bounded Morse recording\n"
        "  stop            stop and save the active recording\n"
        "  strength 1|2|3 set low, medium or high LRA command\n"
        "  status          display logger/parser statistics\n"
        "  help            show this list\n"
        "  quit            close files and exit\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESP32 Morse-vibration serial data logger")
    parser.add_argument("--port", help="Serial port, for example COM5 or /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--dataset", type=Path, default=Path("MorseDataset"))
    parser.add_argument("--participant", help="Participant identifier, for example P01")
    parser.add_argument("--session", help="Session identifier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if PACKET_SIZE != 42:
        raise RuntimeError(f"Packet definition is {PACKET_SIZE} bytes, expected 42")

    layout = DatasetLayout(args.dataset)
    port = choose_port(args.port)
    metadata = collect_session_metadata(args.participant, args.session)
    session_path = write_session_metadata(layout, metadata)
    print(f"Session metadata: {session_path}")

    logger: Optional[SerialLogger] = None
    try:
        print(f"Opening {port} at {args.baud} baud...")
        logger = SerialLogger(port, args.baud, layout, metadata)
        time.sleep(1.0)  # ESP32-S3 may reset when the serial port opens.
        logger.enter_binary_mode()
        logger.set_strength("2")
        print("Binary packet mode enabled.")
        print_commands()

        while True:
            logger.check_reader()
            command = input("logger> ").strip().lower()
            if command in {"calibrate", "cal", "c"}:
                logger.calibrate()
            elif command in {"record", "rec", "start"}:
                logger.start_recording()
            elif command in {"stop", "end"}:
                logger.stop_recording()
            elif command.startswith("strength "):
                logger.set_strength(command.split(maxsplit=1)[1])
            elif command in {"status", "i"}:
                logger.print_status()
            elif command in {"help", "h", "?"}:
                print_commands()
            elif command in {"quit", "exit", "q"}:
                break
            elif not command:
                continue
            else:
                print("Unknown command. Type 'help'.")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nStopping logger...")
        return 0
    except (serial.SerialException, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
