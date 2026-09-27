#!/bin/bash
# Explicit launch after full600 audit and GPU release only.
set -u
cd /data1/matengyu/geyugong/neuroadapter-author-first || exit 1
B=benchmarks/subj01-v1
export CUDA_VISIBLE_DEVICES="${1:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ -e "$B/logs/mindbridge-test100.log" ] || [ -e "$B/logs/mindbridge-score-test100.log" ]; then
  printf '%s\n' 'Existing attempt: inspect logs and outputs before explicit recovery.'
  exit 32
fi
"$B/envs/mindbridge/bin/python" -u "$B/mindbridge_reconstruct_k1.py" --pool test100 --generate > "$B/logs/mindbridge-test100.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$B/logs/mindbridge-test100.exit"
[ "$rc" -eq 0 ] || exit "$rc"
runtime/env/bin/python -u "$B/evaluate_k1.py" --method mindbridge --ids "$B/test100_ids.txt" --feature-batch 16 > "$B/logs/mindbridge-score-test100.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$B/logs/mindbridge-score-test100.exit"
exit "$rc"
