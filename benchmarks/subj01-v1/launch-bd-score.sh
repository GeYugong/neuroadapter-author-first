#!/bin/bash
cd /data1/matengyu/geyugong/neuroadapter-author-first
for pool in test100_ids test_full_ids; do
 CUDA_VISIBLE_DEVICES=0 runtime/env/bin/python -u benchmarks/subj01-v1/evaluate_k1.py --method brain-diffuser --ids "benchmarks/subj01-v1/${pool}.txt" --feature-batch 16 > "benchmarks/subj01-v1/logs/bd-score-${pool}.log" 2>&1
 rc=$?
 printf '%s\n' "$rc" > "benchmarks/subj01-v1/logs/bd-score-${pool}.exit"
 if [ "$rc" != 0 ]; then exit "$rc"; fi
done