#!/usr/bin/env python3
"""Create leakage-group-safe splits from an existing navigation manifest.

Outputs are written to a new directory and manifest.  Existing training split
files are never modified by this tool.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_nav_manifest import FIELDS, ManifestError


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"Manifest missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ManifestError("Manifest has no rows")
    return rows


def assign_groups(
    rows: list[dict[str, str]],
    test_fraction: float,
    val_fraction: float,
    seed: int,
    max_fraction_deviation: float | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not 0 <= test_fraction < 1 or not 0 <= val_fraction < 1:
        raise ManifestError("Fractions must be in [0, 1)")
    if test_fraction + val_fraction >= 1:
        raise ManifestError("test_fraction + val_fraction must be < 1")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = row["leakage_group"].strip()
        if not group:
            raise ManifestError(f"Empty leakage_group for {row['trajectory_id']}")
        groups[group].append(row)

    rng = random.Random(seed)
    shuffled = list(groups.items())
    rng.shuffle(shuffled)
    # Large groups are assigned first; the random tie breaker makes equal-size
    # groups seed-dependent while retaining deterministic output.
    tie = {group: rng.random() for group in groups}
    shuffled.sort(key=lambda item: (-len(item[1]), tie[item[0]]))

    fractions = {
        "train": 1.0 - test_fraction - val_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    active = {name: value for name, value in fractions.items() if value > 0}
    targets = {name: len(rows) * fraction for name, fraction in active.items()}
    counts = {name: 0 for name in active}
    assignment: dict[str, str] = {}

    for group, members in shuffled:
        # Pick the split with the largest normalized remaining deficit.
        def deficit(split: str) -> tuple[float, float, str]:
            target = targets[split]
            remaining = target - counts[split]
            return (remaining / target if target else -1.0, remaining, split)

        chosen = max(active, key=deficit)
        assignment[group] = chosen
        counts[chosen] += len(members)

    updated = []
    for row in rows:
        copy = dict(row)
        copy["split"] = assignment[row["leakage_group"].strip()]
        updated.append(copy)
    updated.sort(key=lambda row: (row["dataset_id"], row["trajectory_id"]))

    actual_fractions = {name: count / len(rows) for name, count in counts.items()}
    if any(count == 0 for count in counts.values()):
        raise ManifestError(f"Grouped assignment produced an empty split: {counts}")
    if max_fraction_deviation is not None:
        deviations = {
            name: abs(actual_fractions[name] - fractions[name]) for name in active
        }
        if max(deviations.values(), default=0.0) > max_fraction_deviation:
            raise ManifestError(
                f"Split fraction deviation exceeds {max_fraction_deviation}: {deviations}"
            )

    session_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in updated:
        session_splits[(row["dataset_id"].strip(), row["source_session"].strip())].add(row["split"])
    leaked_sessions = [session for session, splits in session_splits.items() if len(splits) > 1]
    if leaked_sessions:
        raise ManifestError(
            "leakage_group is finer than source_session; cross-split sessions remain: "
            + repr(leaked_sessions[:10])
        )

    report = {
        "seed": seed,
        "row_count": len(rows),
        "group_count": len(groups),
        "target_fractions": fractions,
        "counts": counts,
        "actual_fractions": actual_fractions,
        "max_fraction_deviation": max_fraction_deviation,
    }
    return updated, report


def write_outputs(rows: list[dict[str, str]], out_manifest: Path, out_splits: Path) -> None:
    if out_manifest.exists() or out_splits.exists():
        raise ManifestError(
            "Refusing to overwrite existing output; choose a new path or remove it explicitly"
        )
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    by_dataset_split: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        by_dataset_split[(row["dataset_id"], row["split"])].append(row["trajectory_id"])
    for (dataset_id, split), names in sorted(by_dataset_split.items()):
        destination = out_splits / dataset_id / split
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "traj_names.txt").write_text(
            "\n".join(sorted(names)) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build group-safe navigation data splits.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-splits", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-fraction-deviation", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows, report = assign_groups(
            load_rows(args.manifest),
            args.test_fraction,
            args.val_fraction,
            args.seed,
            args.max_fraction_deviation,
        )
        write_outputs(rows, args.out_manifest, args.out_splits)
    except (ManifestError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"Wrote {report['row_count']} rows / {report['group_count']} leakage groups; "
        f"counts={report['counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
