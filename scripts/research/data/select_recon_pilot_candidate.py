#!/usr/bin/env python3
"""Select the smallest canonical-ready RECON HDF5 from a complete size prefix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from convert_recon_huron_pilot import sha256_file


def inventory_prefix(inventory: Path, max_bytes: int) -> dict[str, int]:
    expected = {}
    for line in inventory.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if (
            len(fields) == 6
            and fields[5].endswith(".hdf5")
            and int(fields[2]) <= max_bytes
        ):
            expected[fields[5]] = int(fields[2])
    return expected


def select(
    extracted_root: Path,
    inventory: Path,
    max_bytes: int,
    min_frames: int,
) -> dict:
    import h5py

    expected = inventory_prefix(inventory, max_bytes)
    candidates = []
    errors = []
    actual_paths = set()
    for path in sorted(extracted_root.rglob("*.hdf5")):
        relative = path.relative_to(extracted_root).as_posix()
        actual_paths.add(relative)
        try:
            with h5py.File(path, "r") as handle:
                required = ("images/rgb_left", "jackal/position", "jackal/yaw")
                missing = [name for name in required if name not in handle]
                counts = (
                    [int(handle[name].shape[0]) for name in required]
                    if not missing
                    else []
                )
            candidates.append(
                {
                    "relative_path": relative,
                    "bytes": path.stat().st_size,
                    "inventory_bytes": expected.get(relative),
                    "counts": counts,
                    "missing_nodes": missing,
                    "eligible": (
                        not missing
                        and len(set(counts)) == 1
                        and counts[0] >= min_frames
                    ),
                }
            )
        except (OSError, KeyError, ValueError) as exc:
            errors.append({"relative_path": relative, "error": str(exc)})
    missing_extractions = sorted(set(expected) - actual_paths)
    unexpected_extractions = sorted(actual_paths - set(expected))
    size_mismatches = [
        item["relative_path"]
        for item in candidates
        if item["inventory_bytes"] != item["bytes"]
    ]
    eligible = sorted(
        (item for item in candidates if item["eligible"]),
        key=lambda item: (item["bytes"], item["relative_path"]),
    )
    selected = eligible[0] if eligible else None
    if selected:
        selected_path = extracted_root / selected["relative_path"]
        selected = dict(selected)
        selected["sha256"] = sha256_file(selected_path)
    gates = {
        "inventory_prefix_nonempty": bool(expected),
        "complete_prefix_extracted": not missing_extractions,
        "no_unexpected_hdf5": not unexpected_extractions,
        "inventory_sizes_match": not size_mismatches,
        "all_candidates_readable": not errors,
        "eligible_candidate_present": selected is not None,
    }
    return {
        "schema_version": "0.1.0",
        "audit_id": "recon-smallest-canonical-ready-candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Minimum (archive member bytes, relative path) among every HDF5 member "
            f"with archive bytes <= {max_bytes}, equal required stream lengths, and "
            f"at least {min_frames} frames. Because the selected member is within the "
            "complete prefix, no smaller archive member can satisfy the rule."
        ),
        "inventory": {
            "path": inventory.as_posix(),
            "sha256": sha256_file(inventory),
            "max_member_bytes": max_bytes,
            "expected_count": len(expected),
        },
        "extracted_root": extracted_root.as_posix(),
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected": selected,
        "missing_extractions": missing_extractions,
        "unexpected_extractions": unexpected_extractions,
        "size_mismatches": size_mismatches,
        "errors": errors,
        "candidates": candidates,
        "gates": gates,
        "blocking_gates": [name for name, passed in gates.items() if not passed],
        "passed": all(gates.values()),
        "execution_counts": {
            "downloads": 0,
            "semantic_conversions": 0,
            "model_forward": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "training": 0,
            "evaluation": 0,
            "simulation": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=14)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = select(
        args.extracted_root,
        args.inventory,
        args.max_bytes,
        args.min_frames,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"candidate selection passed={report['passed']}, "
        f"candidates={report['candidate_count']}, selected={report['selected']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
