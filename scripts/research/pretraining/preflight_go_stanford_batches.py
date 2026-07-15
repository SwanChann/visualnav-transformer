#!/usr/bin/env python3
"""Run a CPU-only canonical DataLoader preflight without a model forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from go_stanford_adapter import (
    GoStanfordCanonicalDataset,
    make_go_stanford_dataloader,
)


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_context(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def run_preflight(
    *,
    repo_root: Path,
    dataset_root: Path,
    manifest_path: Path,
    data_audit_path: Path,
    smoke_config_path: Path,
    model_contract_path: Path,
    max_batches: int,
) -> dict[str, Any]:
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    smoke = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "planned_no_training_executed" or smoke.get("actual_results") is not None:
        raise ValueError("smoke config must remain unexecuted during DataLoader preflight")
    audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("content-level Go Stanford audit must pass before batch preflight")
    if canonical_text_sha256(manifest_path) != audit["provenance"]["manifest_sha256"]:
        raise ValueError("manifest hash does not match the content-level audit")

    observation_frames = int(smoke["observation_frames"])
    action_history_steps = int(smoke["action_history_steps"])
    action_horizon = int(smoke["action_horizon"])
    data_fraction = float(smoke["data_fraction"])
    batch_size = int(smoke["batch_size_candidate"])
    seed = int(smoke["seed"])
    dataset = GoStanfordCanonicalDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
        observation_frames=observation_frames,
        action_history_steps=action_history_steps,
        action_horizon=action_horizon,
        image_size=(96, 96),
        data_fraction=data_fraction,
        subset_seed=seed,
    )
    loader = make_go_stanford_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    image_values: list[torch.Tensor] = []
    target_norms: list[torch.Tensor] = []
    trajectory_ids: set[str] = set()
    history_valid = 0
    history_total = 0
    samples = 0
    started = time.perf_counter()
    batches = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batches += 1
        batch_samples = int(batch["obs_images"].shape[0])
        samples += batch_samples
        trajectory_ids.update(batch["trajectory_id"])
        image_values.extend([batch["obs_images"].amin(), batch["obs_images"].amax()])
        target_norms.append(torch.linalg.vector_norm(batch["target_waypoints"], dim=-1).reshape(-1))
        history_valid += int(batch["action_history_mask"].sum())
        history_total += int(batch["action_history_mask"].numel())
        if not bool(torch.isfinite(batch["obs_images"]).all()):
            raise ValueError(f"non-finite observation tensor in batch {batch_index}")
        if not bool(torch.isfinite(batch["target_waypoints"]).all()):
            raise ValueError(f"non-finite target tensor in batch {batch_index}")
        if not bool(batch["target_mask"].all()):
            raise ValueError(f"unexpected masked target in fully eligible batch {batch_index}")
    elapsed = time.perf_counter() - started
    if batches < 1 or samples < 1:
        raise ValueError("preflight produced no batches")
    norms = torch.cat(target_norms).numpy()
    image_min = min(float(value) for value in image_values)
    image_max = max(float(value) for value in image_values)
    expected_subset = max(1, int(round(dataset.full_sample_count * data_fraction)))
    commit, dirty = git_context(repo_root)
    adapter_path = Path(__file__).with_name("go_stanford_adapter.py")

    return {
        "schema_version": "0.1.0",
        "preflight_id": "go-stanford-b0-batch-readiness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_boundary": {
            "training_performed": False,
            "model_instantiated": False,
            "model_forward_performed": False,
            "backward_performed": False,
            "optimizer_created": False,
            "gpu_tensor_created": False,
            "dataset_mutated": False,
        },
        "provenance": {
            "git_commit": commit,
            "tracked_worktree_dirty": dirty,
            "manifest_sha256": audit["provenance"]["manifest_sha256"],
            "split_sha256": audit["provenance"]["split_sha256"],
            "dataset_tree_sha256": audit["provenance"]["canonical_dataset_tree_sha256"],
            "adapter_sha256": file_sha256(adapter_path),
            "preflight_script_sha256": file_sha256(Path(__file__)),
            "smoke_config_sha256": canonical_text_sha256(smoke_config_path),
            "model_contract_sha256": canonical_text_sha256(model_contract_path),
        },
        "dataset": {
            "split": "train",
            "full_canonical_sample_count": dataset.full_sample_count,
            "data_fraction": data_fraction,
            "expected_subset_sample_count": expected_subset,
            "actual_subset_sample_count": len(dataset),
            "subset_seed": seed,
        },
        "observed_batches": {
            "requested_max_batches": max_batches,
            "batches": batches,
            "samples": samples,
            "unique_trajectories": len(trajectory_ids),
            "batch_size_candidate": batch_size,
            "elapsed_s": elapsed,
            "samples_per_s_cpu_single_worker": samples / elapsed,
            "observation_tensor_min": image_min,
            "observation_tensor_max": image_max,
            "history_valid_fraction": history_valid / history_total,
            "target_radius_quantiles_m": {
                "p01": float(np.quantile(norms, 0.01)),
                "p50": float(np.quantile(norms, 0.50)),
                "p99": float(np.quantile(norms, 0.99)),
                "max": float(np.max(norms)),
            },
        },
        "gates": {
            "content_audit_passed": True,
            "manifest_identity_matches": True,
            "deterministic_fraction_count_matches": len(dataset) == expected_subset,
            "canonical_batches_passed": True,
            "image_net_normalization_applied": True,
            "full_image_encoder_implemented": False,
            "train_loop_implemented": False,
            "model_forward_authorized": False,
            "b0_training_authorized": False,
        },
        "ready_for_image_encoder_implementation": True,
        "ready_for_b0_training": False,
        "claim_boundary": (
            "This is a CPU DataLoader/canonical-batch readiness check only. It is not a model "
            "forward, loss evaluation, throughput measurement for training, or training result."
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    observed = report["observed_batches"]
    gates = report["gates"]
    lines = [
        "# Go Stanford B0 Batch Readiness",
        "",
        f"- Full train canonical samples: {dataset['full_canonical_sample_count']}",
        f"- Frozen 5% subset: {dataset['actual_subset_sample_count']}",
        f"- Observed batches / samples: {observed['batches']} / {observed['samples']}",
        f"- Unique trajectories observed: {observed['unique_trajectories']}",
        f"- CPU single-worker samples/s: {observed['samples_per_s_cpu_single_worker']:.2f}",
        f"- Ready for image-encoder implementation: {report['ready_for_image_encoder_implementation']}",
        f"- Ready for B0 training: {report['ready_for_b0_training']}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in gates.items())
    lines.extend(["", "## Evidence boundary", "", report["claim_boundary"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--smoke-config", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_preflight(
            repo_root=args.repo_root,
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            data_audit_path=args.data_audit,
            smoke_config_path=args.smoke_config,
            model_contract_path=args.model_contract,
            max_batches=args.max_batches,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)
    print(
        f"Read {report['observed_batches']['batches']} canonical batches / "
        f"{report['observed_batches']['samples']} samples without model execution"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
