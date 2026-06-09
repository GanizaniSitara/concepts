#!/usr/bin/env python
"""
Burn simple TikTok/Reels-style captions into an MP4 with ffmpeg drawtext.

Requirements:
- Python 3.x
- ffmpeg on PATH

Usage:
  python 2026-06_overlay_captions.py input.mp4 2026-06_overlay_captions.example.json output.mp4

Captions JSON schema:
[
  {"start": 0.0, "end": 2.5, "text": "Hook line"},
  {"start": 2.5, "end": 5.0, "text": "Line one\nLine two"}
]

Quick tweaks:
- FONT_SIZE controls text size.
- Y_FRACTION controls vertical placement; 0.08 means the caption block sits
  8% above the bottom edge.
- CRF controls output quality/size. Lower is bigger and cleaner; 18 is a good
  high-quality default for preserving portrait video quality.
- The default style is white text with a black outline. To use a box style,
  replace bordercolor/borderw with box=1, boxcolor=black@0.6, boxborderw=20.

Caption lines are passed to drawtext through temporary text files. That keeps
literal characters such as :, ', comma, and backslash intact across ffmpeg
builds while still using one drawtext filter per visible line.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FONT = "Arial"
FONT_SIZE = 64
FONT_COLOR = "white"
BORDER_COLOR = "black"
BORDER_WIDTH = 4
Y_FRACTION = 0.08
LINE_GAP = 10
VIDEO_CODEC = "libx264"
PRESET = "slow"
CRF = 18


@dataclass(frozen=True)
class Caption:
    start: float
    end: float
    lines: list[str]


@dataclass(frozen=True)
class RenderedLine:
    start: float
    end: float
    line_index: int
    line_count: int
    text_file: Path


def escape_ffmpeg_filter_value(text: str) -> str:
    """Escape characters that are special inside ffmpeg filter option values."""
    return (
        text.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
    )


def format_filter_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def read_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def split_caption_lines(text: str, label: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in normalized.split("\n") if line != ""]
    if not lines:
        raise ValueError(f"{label} text must contain at least one non-empty line")
    return lines


def load_captions(path: Path) -> list[Caption]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("captions JSON must be a list")
    if not payload:
        raise ValueError("captions JSON must contain at least one caption")

    captions: list[Caption] = []
    for index, item in enumerate(payload, start=1):
        label = f"caption {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")

        missing = {"start", "end", "text"} - item.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{label} is missing required field(s): {names}")

        start = read_number(item["start"], f"{label}.start")
        end = read_number(item["end"], f"{label}.end")
        if end <= start:
            raise ValueError(f"{label}.end must be greater than {label}.start")

        text = item["text"]
        if not isinstance(text, str):
            raise ValueError(f"{label}.text must be a string")

        captions.append(Caption(start=start, end=end, lines=split_caption_lines(text, label)))

    return captions


def line_y_expression(line_index: int, line_count: int) -> str:
    line_step = FONT_SIZE + LINE_GAP
    block_height = line_count * FONT_SIZE + max(0, line_count - 1) * LINE_GAP
    line_offset = line_index * line_step
    bottom_anchor = f"h*{format_filter_number(1 - Y_FRACTION)}"
    return f"{bottom_anchor}-{block_height}+{line_offset}"


def escape_textfile_path(path: Path) -> str:
    escaped = escape_ffmpeg_filter_value(path.resolve().as_posix())
    return escaped.replace(r"\:", r"\\:")


def write_caption_text_files(captions: list[Caption], temp_dir: Path) -> list[RenderedLine]:
    rendered_lines: list[RenderedLine] = []
    for caption_index, caption in enumerate(captions, start=1):
        line_count = len(caption.lines)
        for line_index, line in enumerate(caption.lines):
            text_file = temp_dir / f"caption_{caption_index:03d}_line_{line_index + 1:02d}.txt"
            text_file.write_text(line, encoding="utf-8")
            rendered_lines.append(
                RenderedLine(
                    start=caption.start,
                    end=caption.end,
                    line_index=line_index,
                    line_count=line_count,
                    text_file=text_file,
                )
            )
    return rendered_lines


def build_drawtext_filter(lines: list[RenderedLine]) -> str:
    parts: list[str] = []

    for line in lines:
        start = format_filter_number(line.start)
        end = format_filter_number(line.end)
        enable = f"between(t\\,{start}\\,{end})"
        drawtext = (
            f"drawtext=textfile={escape_textfile_path(line.text_file)}"
            f":font={FONT}"
            f":fontcolor={FONT_COLOR}"
            f":fontsize={FONT_SIZE}"
            f":bordercolor={BORDER_COLOR}"
            f":borderw={BORDER_WIDTH}"
            f":x=(w-text_w)/2"
            f":y={line_y_expression(line.line_index, line.line_count)}"
            f":enable={enable}"
        )
        parts.append(drawtext)

    return ",".join(parts)


def run_ffmpeg(input_path: Path, captions_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input video not found: {input_path}")
    if not captions_path.exists():
        raise SystemExit(f"Captions file not found: {captions_path}")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("Output path must be different from input path")
    if not output_path.parent.exists():
        raise SystemExit(f"Output directory not found: {output_path.parent}")

    try:
        captions = load_captions(captions_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Could not load captions: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="overlay_captions_") as temp_name:
        rendered_lines = write_caption_text_files(captions, Path(temp_name))
        vf_filter = build_drawtext_filter(rendered_lines)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            vf_filter,
            "-c:v",
            VIDEO_CODEC,
            "-preset",
            PRESET,
            "-crf",
            str(CRF),
            "-c:a",
            "copy",
            str(output_path),
        ]

        print("Running ffmpeg command:")
        print(subprocess.list2cmdline(cmd))

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise SystemExit("ffmpeg not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"ffmpeg failed with exit code {exc.returncode}") from exc

    print(f"Done. Output written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Burn JSON-timed TikTok/Reels-style captions into an MP4."
    )
    parser.add_argument("input_mp4", type=Path, help="Input MP4 file")
    parser.add_argument("captions_json", type=Path, help="Captions JSON file")
    parser.add_argument("output_mp4", type=Path, help="Output MP4 file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ffmpeg(args.input_mp4, args.captions_json, args.output_mp4)


if __name__ == "__main__":
    main()
