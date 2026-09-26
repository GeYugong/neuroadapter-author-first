#!/bin/bash
cd /data1/matengyu/geyugong/neuroadapter-author-first
CUDA_VISIBLE_DEVICES=0,1 benchmarks/subj01-v1/envs/p1-author310/bin/python -u benchmarks/subj01-v1/brain_diffuser_infer.py --generate > benchmarks/subj01-v1/logs/brain-diffuser-full.log 2>&1
rc=$?
printf '%s\n' "$rc" > benchmarks/subj01-v1/logs/brain-diffuser-full.exit
exit "$rc"