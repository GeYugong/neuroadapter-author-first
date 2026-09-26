#!/bin/bash
cd /data1/matengyu/geyugong/neuroadapter-author-first
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN HOME=/data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/cache/inference-home HF_HOME=/data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/cache/hf-inference HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 /data1/matengyu/geyugong/neuroadapter-author-first/runtime/env/bin/python /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/run_neuroadapter.py --ids /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/test_full_ids.txt > /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/neuroadapter-testfull.log 2>&1
status=$?
printf "%s\n" "$status" > /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/neuroadapter-testfull.exit
[ "$status" = "0" ] || exit "$status"
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 /data1/matengyu/geyugong/neuroadapter-author-first/runtime/env/bin/python /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/evaluate_k1.py --method neuroadapter --ids /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/test_full_ids.txt --feature-batch 16 > /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/neuroadapter-scorefull.log 2>&1
status=$?
printf "%s\n" "$status" > /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/neuroadapter-scorefull.exit
exit "$status"
