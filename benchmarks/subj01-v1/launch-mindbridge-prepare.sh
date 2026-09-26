#!/bin/bash
set -uo pipefail
cd /data1/matengyu/geyugong/neuroadapter-author-first
export CUDA_VISIBLE_DEVICES=""
runtime/env/bin/python -u benchmarks/subj01-v1/prepare_mindbridge_data.py > benchmarks/subj01-v1/logs/mindbridge-data-prepare.log 2>&1
rc=$?
printf '%s\n' "$rc" > benchmarks/subj01-v1/logs/mindbridge-data-prepare.exit
exit "$rc"
