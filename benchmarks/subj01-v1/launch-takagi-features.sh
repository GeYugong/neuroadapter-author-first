#!/bin/bash
cd /data1/matengyu/geyugong/neuroadapter-author-first
CUDA_VISIBLE_DEVICES=1 benchmarks/subj01-v1/envs/p1-author310/bin/python -u benchmarks/subj01-v1/takagi_extract_features.py --extract > benchmarks/subj01-v1/logs/takagi-feature-full.log 2>&1
rc=$?
printf '%s\n' "$rc" > benchmarks/subj01-v1/logs/takagi-feature-full.exit
exit "$rc"