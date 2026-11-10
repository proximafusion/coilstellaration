---
title: Coilstellaration Bench Evaluator
emoji: ⚙️
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: leaderboard/app.py
pinned: false
license: mit
short_description: Scores queued Coilstellaration Bench submissions
datasets:
- proxima-fusion/coilstellaration
- proxima-fusion/coilstellaration-bench-requests
- proxima-fusion/coilstellaration-bench-results
tags:
- evaluator
- coilstellaration
- proxima-fusion
---

# Coilstellaration Bench Evaluator

Worker for the [Coilstellaration Bench](https://huggingface.co/spaces/proxima-fusion/coilstellaration-bench)
leaderboard. It polls
[`coilstellaration-bench-requests`](https://huggingface.co/datasets/proxima-fusion/coilstellaration-bench-requests)
for pending submissions, evaluates them with the coilstellaration physics
pipeline, and pushes scores to
[`coilstellaration-bench-results`](https://huggingface.co/datasets/proxima-fusion/coilstellaration-bench-results).
The page only shows worker status. Run a single evaluator per queue.

Requires the Space variable `COILBENCH_ROLE=runner` and an `HF_TOKEN` secret with
write access to both datasets; see `leaderboard/HOSTING.md`.

This Space is synced from [proximafusion/coilstellaration](https://github.com/proximafusion/coilstellaration);
this README is generated from `leaderboard/space.evaluator.md`, so edit that
file rather than this one.
