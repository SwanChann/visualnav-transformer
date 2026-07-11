# Current RA-L paper package

## Decision

**Conditional / not submission-ready.** The defensible paper is a narrow systems
study of a frozen NoMaD-style policy deployed on Lite3, centered on the latency–
trajectory-proxy trade-off and honest evidence boundaries. It is not a new-policy,
cross-dataset, or state-of-the-art navigation paper.

Fig. 2 and Table II are locally reproducible. Historical MuJoCo records and real
videos are useful integration evidence, but they do not currently support a clean
closed-loop comparison or a real-robot success rate. Submission requires the gates
in `05_reviewer_checklist.md`; local analysis cannot manufacture those experiments.

Reproduce the local evidence with:

```powershell
python scripts/analysis/ral_offline_pareto.py --results results --out 投稿冲刺/workspace/offline_pareto
python scripts/analysis/ral_mujoco_summary.py --results results --out 投稿冲刺/workspace/mujoco
python scripts/analysis/ral_real_robot_timing.py --trial-root history-doc/毕设冲刺/video --out 投稿冲刺/workspace/real_robot
python scripts/analysis/ral_real_video_review.py --trial-root history-doc/毕设冲刺/video --annotation-csv 投稿冲刺/workspace/real_robot/real_trial_annotation_template.csv --out 投稿冲刺/workspace/real_robot/review --samples 12
```

No training, simulation, or robot run was performed while building this package.
