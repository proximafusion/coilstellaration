---
title: Coilstellaration Bench
emoji: 🥇
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: leaderboard/app.py
pinned: true
license: mit
short_description: Benchmark ML models that predict stellarator coilsets
datasets:
- proxima-fusion/coilstellaration
- proxima-fusion/coilstellaration-bench-requests
- proxima-fusion/coilstellaration-bench-results
tags:
- arxiv:2611.12689
- leaderboard
- coilstellaration
- proxima-fusion
---

# Coilstellaration Bench

Leaderboard for ONNX models that predict a stellarator coilset from a plasma
boundary and a set of requirement metrics, accompanying the paper
[CoilStellaration: A Dataset and Benchmark for Engineering-Aware Stellarator Coilset Generation](https://arxiv.org/abs/2611.12689).
The ONNX contract, scoring and citation are in the Space's About tab. Submissions are queued in
[`coilstellaration-bench-requests`](https://huggingface.co/datasets/proxima-fusion/coilstellaration-bench-requests),
scored by the
[evaluator Space](https://huggingface.co/spaces/proxima-fusion/coilstellaration-bench-evaluator),
and published to
[`coilstellaration-bench-results`](https://huggingface.co/datasets/proxima-fusion/coilstellaration-bench-results).

This Space is synced from [proximafusion/coilstellaration](https://github.com/proximafusion/coilstellaration);
this README is generated from `leaderboard/space.leaderboard.md`, so edit that
file rather than this one.
