# Real-robot video review

This directory is an evidence-review aid, not a source of automatically assigned
success labels.

- `video_review_manifest.csv` inventories 20 videos from 14 historical trials.
- `contact_sheets/` contains 12 uniformly sampled frames per video. Original videos
  remain untouched.
- `ai_assisted_trial_review.csv` records only visually supportable observations.

All 14 `formal_outcome` values remain `needs_human_goal_check`. Directory names such
as “成功” and “失败” are treated as unverified priors because the frozen goal pose,
success radius, timeout, intervention record, and terminal trigger are absent from
the videos. Do not derive a real-robot success rate from this review. The canonical
manual entry point remains `../real_trial_annotation_template.csv`.
