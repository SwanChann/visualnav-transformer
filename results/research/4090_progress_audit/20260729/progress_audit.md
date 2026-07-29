# RTX 4090 server progress audit — 2026-07-29

## Verdict

The governed 4090 work through the RECON/HuRoN conversion pilot is preserved and
auditable. Go Stanford DATA-PILOT, TinyNavBrain IMAGE-FORWARD, the two-step
TRAIN-STEP gate, the H0/H1 B0 plumbing smoke, and the two isolated conversion
pilots are complete within their stated evidence boundaries.

This is not a completed multi-dataset training result. RECON/HuRoN are not ready
for promotion, no B0 checkpoint has been evaluated, and no baseline matrix,
multi-seed H1 training, cross-dataset offline gate, promoted simulation, or new
real-robot experiment has been run.

The current live server state adds one execution blocker: the NVIDIA kernel
modules are loaded, but `/dev/nvidia*` is absent. `nvidia-smi` cannot communicate
with the driver and PyTorch reports CUDA unavailable with zero visible devices.
Historical GPU receipts remain valid evidence of the executions recorded on
2026-07-15/16, but they do not establish that GPUs are currently usable.

## Verified facts

### Git and worktree

- Branch: `agent/ubuntu-sim-handoff`
- Audited pre-report HEAD and upstream:
  `c956a91fa347822a747d50bd55d9b2b1a77f608a`
- Tracked worktree changes before this report: 0
- Preserved untracked paths: 3,800
  - 3,697 under `nomad_dataset/`, including 3,696 metadata `.backup` files and
    one Go Stanford archive
  - 87 W&B paths
  - 12 historical result paths
  - two historical training helper/config paths
  - two personal helper/document paths
- None of those untracked paths belongs to this audit commit. No data,
  checkpoint, W&B payload, old result, or personal helper is to be staged.

### Current environment

- OS kernel: Linux 6.8.0-124-generic, x86_64
- Conda environment: `nomad_train`
- Python / PyTorch / torchvision: 3.8.5 / 2.4.1+cu121 / 0.19.1+cu121
- PyTorch CUDA build: 12.1
- Root filesystem: 3.7 TiB total, 3.0 TiB used, 576 GiB available, 84% used
- NVIDIA kernel module version: 580.173.02
- Current GPU live gate: failed
  - `nvidia-smi`: driver communication failure
  - `/dev/nvidia*`: absent
  - `torch.cuda.is_available()`: false
  - visible CUDA devices: 0

### Data

- Go Stanford remains present at `nomad_dataset/go_stanford` (about 1.1 GiB).
  The authoritative content audit records 3,696 trajectories, 198,126 readable
  images, 131,950 canonical train samples, and a frozen 5% subset of 6,598.
  The full image decode was not repeated in this snapshot.
- RECON official archive remains present:
  - bytes: 53,235,196,027
  - recorded SHA-256: `65e32b9902bc9458b9a6da58257b53de8634261e41d256f0b840e7604988c299`
- HuRoN pilot bag remains present:
  - bytes: 11,044,659
  - recorded SHA-256: `71ac9844157bee7e90f9a6ab1d91af10ebad7e39073fd2885ccabc6d66034397`
- The independent processed audits were rerun read-only:
  - RECON: passed, 1 trajectory / 14 frames / 1 canonical window
  - HuRoN/SACSoN: passed, 1 trajectory / 66 frames / 53 canonical windows
  - model, training, evaluation, and simulation executed by these audits: false

### Model and training plumbing

- IMAGE-FORWARD historical gate: passed on GPU 0 with a randomly initialized,
  shared EfficientNet-B0 image policy containing 8,513,117 trainable parameters.
  It is interface/timing evidence only.
- TRAIN-STEP historical gate: two optimizer/backward steps total, exact-resume
  verified. The local integration checkpoint remains present at 102,542,746
  bytes.
- B0 historical gate:
  - H0: 200 optimizer steps / 400 backward calls / 1,600 examples
  - H1: 200 optimizer steps / 400 backward calls / 1,600 examples
  - total: 400 optimizer steps / 800 backward calls
  - six retained local checkpoints: 807,960,660 bytes
  - no evaluation and no pretrained-weight download
- The B0 receipt/checkpoint audit was rerun read-only on 2026-07-29. All gates
  passed, including config/runner identity, finite records, no GradScaler skip,
  exact-resume, retention, receipt hashes, and the authorized space cap.
- `train/logs` occupies about 142 GiB. It is historical local state and still
  lacks a complete per-checkpoint provenance inventory; it is not counted as
  evidence for the TinyNavBrain B0 gate.

## Completed stages

1. 4090 environment and data snapshot (historical Phase 0).
2. Full Go Stanford single-domain content and canonical adapter pilot.
3. TinyNavBrain random-initialized image forward integration.
4. H0/H1 loss, train-step, checkpoint, and exact-resume integration.
5. Go Stanford H0-200 and H1-200 B0 plumbing smoke.
6. Governed RECON/HuRoN acquisition, receipt/checksum/license snapshots.
7. Isolated RECON/HuRoN conversion pilots and train-only group-safe manifest.

## Partial or blocked stages

- **GPU live availability:** blocked now; restore NVIDIA device nodes/driver
  communication before any new CUDA execution.
- **RECON promotion:** blocked by missing per-frame timestamps, kinematically
  inferred `dt`, and absent collection policy/version metadata.
- **HuRoN promotion:** blocked by absent collection policy/version metadata.
- **Evaluation split:** conversion pilots have a train-only manifest and no
  independent holdout.
- **Go Stanford provenance:** raw receipt, observable per-frame timestamps,
  pinned processor version, and independent physical calibration remain absent.
- **Multi-dataset baseline/H1:** not executed.
- **Offline promotion gate for new checkpoints:** not executed.
- **New non-saturated Ubuntu simulation:** not executed.
- **Target-device sustained timing and protocolized real-robot outcomes:** not
  executed.
- **Privacy/consent gate and final RA-L submission gate:** not closed.

## Verification performed in this audit

- Read-only Git/worktree inventory.
- Read-only OS, disk, Python, PyTorch, CUDA, kernel module, and device-node
  snapshot.
- Read-only B0 execution/checkpoint audit.
- Read-only independent RECON and HuRoN processed-pilot audits.
- No model forward, backward, optimizer step, training, evaluation, simulation,
  data conversion, data download, or checkpoint write was performed.

## Promotion decision

The repository and existing receipts are ready to be published as a progress
snapshot. New GPU work is **NO-GO** until the live NVIDIA device failure is
resolved. Multi-dataset training is additionally **NO-GO** until the
RECON/HuRoN metadata and holdout decisions are explicitly resolved or accepted
as documented limitations.
