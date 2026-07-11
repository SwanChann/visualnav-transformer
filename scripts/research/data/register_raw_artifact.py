#!/usr/bin/env python3
"""Create an immutable receipt for an externally acquired dataset artifact.

This tool never downloads or modifies the artifact. It binds a local file to the
official dataset registry, an operator-supplied source URL, and a license snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ALLOWED_LICENSE_STATUSES = {"official_page_explicit", "official_dataset_page_explicit"}
ALLOWED_REDISTRIBUTION_PREFIXES = ("permitted", "share_alike")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(registry_path: Path, dataset_id: str) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [d for d in registry.get("datasets", []) if d.get("dataset_id") == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"dataset_id must match exactly one registry entry: {dataset_id}")
    return matches[0]


def canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_provenance(cwd: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD").lower()
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit if COMMIT_RE.fullmatch(commit) else None, "dirty": dirty}


def check_existing_receipts(directory: Path, dataset_id: str, artifact_sha: str, registry_sha: str) -> None:
    for candidate in directory.glob("*.json") if directory.is_dir() else []:
        try:
            receipt = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if receipt.get("dataset_id") != dataset_id:
            continue
        if receipt.get("artifact", {}).get("sha256") != artifact_sha:
            continue
        if receipt.get("registry_sha256") != registry_sha:
            raise ValueError(
                f"artifact already registered under a different registry hash: {candidate}"
            )


def register(
    registry_path: Path,
    dataset_id: str,
    artifact_path: Path,
    source_url: str,
    license_snapshot: Path,
    license_name: str,
    operator: str,
    output_path: Path,
) -> dict:
    dataset = load_dataset(registry_path, dataset_id)
    if dataset.get("license_status") not in ALLOWED_LICENSE_STATUSES or not str(
        dataset.get("redistribution", "")
    ).startswith(ALLOWED_REDISTRIBUTION_PREFIXES):
        raise ValueError(f"dataset {dataset_id} is blocked by unresolved payload license")
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if not license_snapshot.is_file() or license_snapshot.stat().st_size == 0:
        raise ValueError("a non-empty license snapshot is required")
    declared_license = dataset.get("license_spdx_or_name")
    if license_name != declared_license:
        raise ValueError(f"license name does not match registry: expected {declared_license}")
    parsed_source = urlparse(source_url)
    if parsed_source.scheme != "https" or not parsed_source.netloc:
        raise ValueError("source_url must be an HTTPS URL")
    official_urls = [dataset.get("official_download"), dataset.get("official_page")]
    official_hosts = {urlparse(url).netloc.lower() for url in official_urls if url}
    if parsed_source.netloc.lower() not in official_hosts:
        raise ValueError(f"source_url host is not an official registry host: {parsed_source.netloc}")
    if not operator.strip():
        raise ValueError("operator must be non-empty")
    if output_path.exists():
        raise FileExistsError(f"immutable receipt already exists: {output_path}")
    registry_sha = sha256_file(registry_path)
    artifact_sha = sha256_file(artifact_path)
    check_existing_receipts(output_path.parent, dataset_id, artifact_sha, registry_sha)
    receipt = {
        "schema_version": "0.1.0",
        "dataset_id": dataset_id,
        "display_name": dataset.get("display_name"),
        "registry_sha256": registry_sha,
        "registry_entry_sha256": canonical_sha256(dataset),
        "artifact": {
            "path": artifact_path.as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": artifact_sha,
        },
        "source_url": source_url,
        "official_page": dataset.get("official_page"),
        "license": {
            "name": dataset.get("license_spdx_or_name"),
            "status": dataset.get("license_status"),
            "snapshot_path": license_snapshot.as_posix(),
            "snapshot_sha256": sha256_file(license_snapshot),
            "snapshot_match": "asserted_by_operator",
        },
        "operator": operator.strip(),
        "git": git_provenance(registry_path.parent),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversion_status": "not_started",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-snapshot", type=Path, required=True)
    parser.add_argument("--license-name", required=True)
    parser.add_argument("--operator", default=os.environ.get("USERNAME") or os.environ.get("USER"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    register(
        args.registry,
        args.dataset_id,
        args.artifact,
        args.source_url,
        args.license_snapshot,
        args.license_name,
        args.operator or "",
        args.out,
    )
    print(args.out)


if __name__ == "__main__":
    main()
