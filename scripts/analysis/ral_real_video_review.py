#!/usr/bin/env python3
"""Inventory frozen real-robot videos and render review contact sheets.

Original videos are opened read-only.  Contact sheets are review aids and are
not experimental results or proof of navigation success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VideoReviewError(ValueError):
    pass


def safe_id(trial_id: str, video: Path) -> str:
    digest = hashlib.sha256(video.as_posix().encode("utf-8")).hexdigest()[:10]
    ascii_hint = "navigation" if video.name == "navigation_record.mp4" else "external"
    return f"{digest}_{ascii_hint}"


def inspect_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReviewError(f"Cannot open video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise VideoReviewError(f"Invalid video metadata: {path}")
    return {
        "frame_count": frames,
        "fps": fps,
        "duration_s": frames / fps,
        "width": width,
        "height": height,
    }


def render_contact_sheet(path: Path, output: Path, samples: int = 12, columns: int = 4) -> None:
    meta = inspect_video(path)
    capture = cv2.VideoCapture(str(path))
    positions = np.linspace(0.03, 0.97, samples)
    tile_w, tile_h = 320, 200
    tiles = []
    for fraction in positions:
        frame_index = min(meta["frame_count"] - 1, int(round(fraction * (meta["frame_count"] - 1))))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            label = f"READ ERROR frame={frame_index}"
        else:
            frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            label = f"t={frame_index / meta['fps']:.1f}s  f={frame_index}"
        cv2.rectangle(frame, (0, 0), (tile_w, 24), (0, 0, 0), thickness=-1)
        cv2.putText(frame, label, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(frame)
    capture.release()
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", sheet)
    if not ok:
        raise VideoReviewError(f"Cannot encode contact sheet: {output}")
    output.write_bytes(encoded.tobytes())


def run(trial_root: Path, annotation_csv: Path, out: Path, samples: int) -> list[dict[str, Any]]:
    with annotation_csv.open(encoding="utf-8", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    rows = []
    for annotation in annotations:
        trial_dir = Path(annotation["trial_path"])
        videos = sorted(
            path for path in trial_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        for video in videos:
            meta = inspect_video(video)
            review_id = safe_id(annotation["trial_id"], video)
            sheet = out / "contact_sheets" / f"{review_id}.jpg"
            render_contact_sheet(video, sheet, samples=samples)
            rows.append(
                {
                    "review_id": review_id,
                    "trial_id": annotation["trial_id"],
                    "map": annotation["map"],
                    "video_role": "robot_camera" if video.name == "navigation_record.mp4" else "external_camera",
                    "video_path": video.as_posix(),
                    "contact_sheet": sheet.as_posix(),
                    **meta,
                    "review_status": "pending",
                    "provisional_observation": "",
                    "confidence": "",
                }
            )
    if not rows:
        raise VideoReviewError("No trial videos found")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "video_review_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create read-only video review contact sheets.")
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = run(args.trial_root, args.annotation_csv, args.out, args.samples)
    except (OSError, VideoReviewError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"Wrote {len(rows)} video review rows across "
        f"{len({row['trial_id'] for row in rows})} trials to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
