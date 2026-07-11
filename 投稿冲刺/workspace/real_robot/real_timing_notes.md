# Real-robot Timing Evidence Notes

- Timing files: 7
- Profiled loop samples: 528
- Frozen trial folders with interactive summaries: 14
- Trials with invalid 1970 wall-clock timestamps: ['ddim2cfg0tts8', 'ddim2cfg2tts8', 'ddim3cfg0tts0', 'ddpm', 'pingpong-road']
- Timing trials whose exact inference configuration is not encoded in the log/path: ['0428下午基线，后方走廊，失败', '0428下午最优方案，后方走廊。成功', 'pingpong-road']

## Interpretation boundary

- `total_ms` is the measured profiled loop for that sampled tick; `loop_fps=1000/total_ms`.
- p50/p90/p95 are across sampled individual loop records, not across trial means.
- The CSV is sampled at `profile_interval`, so it may miss unprofiled spikes and is not a complete tail-latency trace.
- Absolute timestamps beginning in 1970 are invalid due to the robot clock; duration fields remain usable if the profiler clock was monotonic.
- Directory names matching `ddim<steps>cfg<weight>tts<budget>` are parsed as configuration metadata. Other labels remain `unknown`; no configuration is guessed from words such as baseline/optimal.
- `interactive_summary.txt status=success` is copied only as `auto_status_untrusted`. It must not populate Table IV until a human reviews the video.
- One timing file is one frozen execution folder, not a randomized repeated trial. Do not calculate success rate from timing-file count.
