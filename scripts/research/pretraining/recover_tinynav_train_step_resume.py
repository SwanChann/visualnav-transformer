#!/usr/bin/env python3
"""Resume-only verification for an existing two-step TinyNavBrain checkpoint.

This recovery path never calls backward or optimizer.step.  It exists so a
post-checkpoint verification failure can be repaired without exceeding the
authorized optimizer-step budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "scripts" / "research" / "models"
sys.path.insert(0, str(MODELS))

from go_stanford_adapter import (  # noqa: E402
    GoStanfordCanonicalDataset,
    collate_go_stanford_canonical,
)
from preflight_tinynav_train_step import (  # noqa: E402
    batch_identity,
    canonical_text_sha256,
    file_sha256,
    git_context,
    move_batch,
    synthetic_batch,
    tensor_state_sha256,
)
from train_step_runtime import (  # noqa: E402
    DeterministicBatchCursor,
    compute_action_loss,
    load_exact_resume_checkpoint,
    restore_rng_state,
)
from tinynavbrain_image_policy import TinyNavBrainImagePolicy  # noqa: E402
from tinynavbrain_scaffold import TinyNavConfig  # noqa: E402


def nested_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda entry: repr(entry)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def rng_probe(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch_cpu": torch.rand(8),
        "torch_cuda": torch.rand(8, device=device),
    }


def rng_probe_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["python"] == right["python"]
        and left["numpy"] == right["numpy"]
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and torch.equal(left["torch_cuda"], right["torch_cuda"])
    )


def optimizer_step_evidence(
    model: TinyNavBrainImagePolicy, optimizer_state: Mapping[str, Any]
) -> dict[str, Any]:
    names = [name for name, _ in model.named_parameters()]
    state = optimizer_state["state"]
    step_by_name: dict[str, int] = {}
    finite = True
    for index, values in state.items():
        name = names[int(index)]
        step = values.get("step", 0)
        step_by_name[name] = int(step.item() if isinstance(step, torch.Tensor) else step)
        for value in values.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                finite = False
    deterministic = {
        name: step for name, step in step_by_name.items()
        if name.startswith("policy.deterministic_head")
    }
    flow = {
        name: step for name, step in step_by_name.items()
        if name.startswith("policy.flow_head") or name.startswith("policy.time_embedding")
    }
    shared = {
        name: step for name, step in step_by_name.items()
        if name.startswith("image_encoder")
    }
    return {
        "finite_optimizer_tensors": finite,
        "deterministic_head_step_values": sorted(set(deterministic.values())),
        "flow_head_step_values": sorted(set(flow.values())),
        "shared_encoder_step_values": sorted(set(shared.values())),
        "optimizer_state_parameter_count": len(state),
        "proves_h0_path_stepped_once": bool(deterministic) and set(deterministic.values()) == {1},
        "proves_h1_path_stepped_once": bool(flow) and set(flow.values()) == {1},
        "proves_shared_path_stepped_twice": bool(shared) and set(shared.values()) == {2},
    }


def run_recovery(
    *,
    repo_root: Path,
    dataset_root: Path,
    manifest_path: Path,
    data_audit_path: Path,
    smoke_config_path: Path,
    model_contract_path: Path,
    checkpoint_path: Path,
    device_index: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
        raise ValueError(f"CUDA device {device_index} is unavailable")
    device = torch.device(f"cuda:{device_index}")
    smoke = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))
    audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "planned_no_training_executed" or smoke.get("actual_results") is not None:
        raise ValueError("frozen B0 config must remain unexecuted")
    if not audit.get("passed"):
        raise ValueError("Go Stanford audit has not passed")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = raw["metadata"]
    extra = raw["extra_state"]
    commit, tracked_dirty, any_dirty = git_context(repo_root)
    expected_metadata = {
        "schema_version": "0.1.0",
        "run_id": "tinynavbrain-train-step-gate-v0.1",
        "phase": "smoke",
        "head": "h1_rectified_flow",
        "seed": int(smoke["seed"]),
        "optimizer_step": 2,
        "git_commit": commit,
        "manifest_sha256": audit["provenance"]["manifest_sha256"],
        "split_sha256": audit["provenance"]["split_sha256"],
        "model_contract_sha256": canonical_text_sha256(model_contract_path),
        "config_sha256": canonical_text_sha256(smoke_config_path),
        "rng_state_present": True,
        "sampler_state_present": True,
    }
    if metadata != expected_metadata:
        raise ValueError("checkpoint metadata does not exactly match current frozen inputs")
    if extra != {
        "gate_not_b0": True,
        "optimizer_steps_executed": 2,
        "heads": ["h0_deterministic", "h1_rectified_flow"],
    }:
        raise ValueError("checkpoint execution boundary is missing or unexpected")

    seed = int(smoke["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = TinyNavConfig()
    dataset = GoStanfordCanonicalDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
        observation_frames=int(smoke["observation_frames"]),
        action_history_steps=int(smoke["action_history_steps"]),
        action_horizon=int(smoke["action_horizon"]),
        image_size=(96, 96),
        data_fraction=float(smoke["data_fraction"]),
        subset_seed=seed,
    )
    initial_model = TinyNavBrainImagePolicy(config, encoder_weights=None).to(device)
    initial_hash = tensor_state_sha256(initial_model.state_dict())
    checkpoint_hash = tensor_state_sha256(raw["model_state"])
    evidence = optimizer_step_evidence(initial_model, raw["optimizer_state"])

    expected_cursor = DeterministicBatchCursor(
        len(dataset), int(smoke["batch_size_candidate"]), seed
    )
    expected_cursor.load_state_dict(raw["sampler_state"])
    expected_indices = expected_cursor.next_indices()
    expected_batch = collate_go_stanford_canonical([dataset[index] for index in expected_indices])
    expected_identity = batch_identity(expected_batch)
    restore_rng_state(raw["rng_state"])
    expected_rng = rng_probe(device)

    resumed_model = TinyNavBrainImagePolicy(config, encoder_weights=None).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-4)
    resumed_cursor = DeterministicBatchCursor(
        len(dataset), int(smoke["batch_size_candidate"]), seed
    )
    loaded = load_exact_resume_checkpoint(
        checkpoint_path,
        expected_metadata=expected_metadata,
        model=resumed_model,
        optimizer=resumed_optimizer,
        sampler=resumed_cursor,
        map_location=device,
    )
    resumed_indices = resumed_cursor.next_indices()
    resumed_batch = collate_go_stanford_canonical([dataset[index] for index in resumed_indices])
    resumed_identity = batch_identity(resumed_batch)
    resumed_rng = rng_probe(device)

    first_real_indices = raw["sampler_state"]["order"][: int(smoke["batch_size_candidate"])]
    first_real_cpu = collate_go_stanford_canonical(
        [dataset[int(index)] for index in first_real_indices]
    )
    with torch.inference_mode():
        resumed_model.eval()
        h0_loss, _ = compute_action_loss(
            resumed_model, synthetic_batch(device, config, seed + 100),
            head="h0_deterministic",
        )
        flow_generator = torch.Generator(device=device).manual_seed(seed + 200)
        h1_loss, _ = compute_action_loss(
            resumed_model, move_batch(first_real_cpu, device),
            head="h1_rectified_flow", generator=flow_generator,
        )
    postcheckpoint_losses = {
        "synthetic_h0": float(h0_loss),
        "real_h1": float(h1_loss),
        "measurement_scope": "post-checkpoint forward-only; original step losses were not retained",
    }
    gates = {
        "checkpoint_records_exactly_two_steps": metadata["optimizer_step"] == 2,
        "checkpoint_explicitly_not_b0": extra["gate_not_b0"] is True,
        "h0_optimizer_path_evidenced": evidence["proves_h0_path_stepped_once"],
        "h1_optimizer_path_evidenced": evidence["proves_h1_path_stepped_once"],
        "shared_encoder_two_step_evidenced": evidence["proves_shared_path_stepped_twice"],
        "finite_optimizer_state": evidence["finite_optimizer_tensors"],
        "model_differs_from_seeded_initialization": initial_hash != checkpoint_hash,
        "model_state_exact_after_reload": (
            checkpoint_hash == tensor_state_sha256(resumed_model.state_dict())
        ),
        "optimizer_state_exact_after_reload": (
            nested_digest(raw["optimizer_state"])
            == nested_digest(resumed_optimizer.state_dict())
        ),
        "rng_state_exact_after_reload": rng_probe_equal(expected_rng, resumed_rng),
        "next_batch_exact_after_reload": (
            expected_indices == resumed_indices and expected_identity == resumed_identity
        ),
        "finite_postcheckpoint_forward_losses": bool(
            np.isfinite([postcheckpoint_losses["synthetic_h0"], postcheckpoint_losses["real_h1"]]).all()
        ),
        "reload_did_not_change_metadata": loaded["metadata"] == metadata,
    }
    if not all(gates.values()):
        raise ValueError(f"resume-only gate failed: {gates}")
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "0.1.0",
        "gate_id": "tinynavbrain-train-step-resume-only-recovery",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_boundary": {
            "checkpoint_recorded_backward_calls": 2,
            "checkpoint_recorded_optimizer_steps": 2,
            "recovery_backward_calls": 0,
            "recovery_optimizer_steps": 0,
            "total_backward_optimizer_steps_across_gate": 2,
            "b0_200_step_run_executed": False,
            "evaluation_executed": False,
            "pretrained_weights_downloaded": False,
        },
        "failure_and_recovery": {
            "original_failure_stage": "post-checkpoint_rng_restore",
            "root_cause": "map_location moved CPU RNG ByteTensor to CUDA",
            "repair": "normalize Torch CPU/CUDA RNG state tensors to CPU before setters",
            "original_step_loss_values_retained": False,
            "training_steps_replayed": False,
        },
        "provenance": {
            "git_commit": commit,
            "tracked_worktree_dirty": tracked_dirty,
            "worktree_dirty_including_preserved_untracked": any_dirty,
            "manifest_sha256": metadata["manifest_sha256"],
            "split_sha256": metadata["split_sha256"],
            "dataset_tree_sha256": audit["provenance"]["canonical_dataset_tree_sha256"],
            "model_contract_sha256": metadata["model_contract_sha256"],
            "smoke_config_sha256": metadata["config_sha256"],
            "runtime_sha256": file_sha256(Path(__file__).with_name("train_step_runtime.py")),
            "writer_script_sha256_current": file_sha256(
                Path(__file__).with_name("preflight_tinynav_train_step.py")
            ),
            "recovery_script_sha256": file_sha256(Path(__file__)),
        },
        "hardware": {
            "device_index": device_index,
            "name": properties.name,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "optimizer_step_evidence": evidence,
        "postcheckpoint_forward_losses": postcheckpoint_losses,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "metadata": metadata,
            "expected_next_batch_identity": expected_identity,
            "resumed_next_batch_identity": resumed_identity,
        },
        "gates": gates,
        "ready_for_b0_execution": False,
        "remaining_before_b0": [
            "review and freeze the TRAIN-STEP implementation/recovery report in Git",
            "obtain separate explicit authorization for the frozen 200-step B0 run",
        ],
        "claim_boundary": (
            "The checkpoint proves two optimizer-path integration steps and exact resume. "
            "Original step loss values were not retained; reported losses are post-checkpoint "
            "forward-only checks. This is not B0, convergence, quality, or method evidence."
        ),
    }


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    losses = report["postcheckpoint_forward_losses"]
    checkpoint = report["checkpoint"]
    lines = [
        "# TinyNavBrain TRAIN-STEP Resume-Only Recovery",
        "",
        f"- GPU: {report['hardware']['name']} (cuda:{report['hardware']['device_index']})",
        "- Total backward/optimizer steps across the gate: 2",
        "- Recovery backward/optimizer steps: 0",
        "- Original numeric step losses retained: False",
        f"- Post-checkpoint forward-only H0 loss: {losses['synthetic_h0']:.8f}",
        f"- Post-checkpoint forward-only H1 loss: {losses['real_h1']:.8f}",
        f"- Checkpoint: `{checkpoint['path']}` ({checkpoint['size_bytes'] / (1024 ** 2):.2f} MiB)",
        f"- Checkpoint SHA-256: `{checkpoint['sha256']}`",
        f"- Ready for B0 execution: {report['ready_for_b0_execution']}",
        "",
        "## Failure and repair",
        "",
        "The two authorized steps and checkpoint write completed. The first reload failed because",
        "`map_location=cuda:0` moved a CPU RNG ByteTensor to CUDA. Recovery normalizes RNG state",
        "to CPU and performs resume-only verification without replaying either training step.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in report["gates"].items())
    lines.extend(["", "## Remaining before B0", ""])
    lines.extend(f"- {item}" for item in report["remaining_before_b0"])
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_recovery(
            repo_root=args.repo_root,
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            data_audit_path=args.data_audit,
            smoke_config_path=args.smoke_config,
            model_contract_path=args.model_contract,
            checkpoint_path=args.checkpoint,
            device_index=args.device_index,
        )
    except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)
    print(
        "Resume-only recovery passed; total optimizer steps remain "
        f"{report['execution_boundary']['total_backward_optimizer_steps_across_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
