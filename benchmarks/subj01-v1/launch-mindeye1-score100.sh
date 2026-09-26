#!/bin/bash
cd /data1/matengyu/geyugong/neuroadapter-author-first
while [ ! -f /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/mindeye1-test100.exit ]; do sleep 5; done
[ "$(cat /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/mindeye1-test100.exit)" = "0" ] || exit 1
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 /data1/matengyu/geyugong/neuroadapter-author-first/runtime/env/bin/python /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/evaluate_k1.py --method mindeye1 --ids /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/test100_ids.txt > /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/mindeye1-score100.log 2>&1
status=$?
printf "%s\n" "$status" > /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/mindeye1-score100.exit
exit "$status"
