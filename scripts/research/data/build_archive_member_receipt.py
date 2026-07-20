#!/usr/bin/env python3
"""Bind one extracted raw pilot file to an immutable parent archive receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_member_size(inventory: Path, member: str) -> int:
    matches = []
    for line in inventory.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == member:
            matches.append(int(fields[2]))
    if len(matches) != 1:
        raise ValueError(f"archive member must occur exactly once in inventory: {member}")
    return matches[0]


def build_receipt(
    dataset_id: str,
    parent_receipt_path: Path,
    inventory_path: Path,
    member: str,
    artifact_path: Path,
    operator: str,
) -> dict:
    parent = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
    if parent.get("dataset_id") != dataset_id:
        raise ValueError("parent receipt dataset_id mismatch")
    if parent.get("conversion_status") != "not_started":
        raise ValueError("parent receipt does not preserve the raw boundary")
    expected_bytes = inventory_member_size(inventory_path, member)
    actual_bytes = artifact_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"extracted member size mismatch: {actual_bytes} != {expected_bytes}"
        )
    if not operator.strip():
        raise ValueError("operator must be non-empty")
    return {
        "schema_version": "0.1.0",
        "dataset_id": dataset_id,
        "receipt_type": "raw_archive_member_extraction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": operator.strip(),
        "parent_archive_receipt": {
            "path": parent_receipt_path.as_posix(),
            "sha256": sha256_file(parent_receipt_path),
            "archive": parent["artifact"],
        },
        "archive_inventory": {
            "path": inventory_path.as_posix(),
            "sha256": sha256_file(inventory_path),
        },
        "archive_member": member,
        "artifact": {
            "path": artifact_path.as_posix(),
            "bytes": actual_bytes,
            "sha256": sha256_file(artifact_path),
        },
        "extraction_status": "selected_raw_member_only",
        "conversion_status": "not_started",
        "execution_counts": {
            "semantic_conversions": 0,
            "model_forward": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "training": 0,
            "evaluation": 0,
            "simulation": 0,
        },
        "claim_boundary": (
            "Lineage and byte-integrity record for one raw archive member only. "
            "The parent archive identity is inherited from its immutable receipt; "
            "this record does not claim full-dataset extraction or semantic conversion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable receipt already exists: {args.out}")
    report = build_receipt(
        args.dataset_id,
        args.parent_receipt,
        args.inventory,
        args.member,
        args.artifact,
        args.operator,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
