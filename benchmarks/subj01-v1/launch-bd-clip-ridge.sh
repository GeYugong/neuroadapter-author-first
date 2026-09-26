#!/bin/bash
cd /data1/matengyu/geyugong/neuroadapter-author-first
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for kind in vision text; do
 benchmarks/subj01-v1/envs/p1-author310/bin/python -u benchmarks/subj01-v1/run_bd_clip_ridge.py --kind "$kind" > "benchmarks/subj01-v1/logs/bd-clip${kind}-ridge.log" 2>&1
 rc=$?
 printf '%s\n' "$rc" > "benchmarks/subj01-v1/logs/bd-clip${kind}-ridge.exit"
 if [ "$rc" != 0 ]; then exit "$rc"; fi
done