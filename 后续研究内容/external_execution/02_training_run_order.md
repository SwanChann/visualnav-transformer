# Stage 2 — training run order

Execution environment: remote RTX 4090 server only. All raw/processed datasets remain
on that server. Windows supplies the frozen Git/config/contract version and receives
small provenance/results; Ubuntu does not run this stage.

Before step 1, record a live server snapshot: full Git SHA, dirty state, Python/CUDA/
PyTorch versions, GPU/VRAM, free disk, dataset inventory, checkpoint hashes and the
exact manifest/split hashes. Repository plans are not proof that the server is ready.

Freeze one manifest/split version, encoder, image resolution, action horizon,
normalization, optimizer budget, parameter tolerance, candidate K and hardware.

1. B0: one specialist per dataset.
2. B1: proportional mixture baseline.
3. B2: balanced mixture baseline.
4. B3: sampler-matched NoMaD/DDPM and DDIM-2/3 controls.
5. B4: deterministic H0 with the same fusion/history interface as H1.
6. H1: TinyNavBrain rectified-flow head at NFE 1/2/4/8.

Use at least three seeds. First run a smoke test that verifies checkpoint reload,
finite losses, output scale, and benchmark JSON validation. Only then run the frozen
budget. Report all seeds, including failures. Primary selection uses validation data;
IID test, corruption and LODO remain sealed until selection is frozen.

Promotion requires H1 to improve a predeclared worst-domain or LODO metric over both
B2/B3 and B4 without a material regression, while remaining non-dominated in
latency. If B4 matches H1, stop generative-head expansion.
