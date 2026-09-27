#!/bin/bash
# Explicit launch after full600 audit and GPU release only.
set -u
cd /data1/matengyu/geyugong/neuroadapter-author-first || exit 1
B=benchmarks/subj01-v1
export CUDA_VISIBLE_DEVICES="${1:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ -e "$B/logs/mindbridge-testfull.log" ] || [ -e "$B/logs/mindbridge-score-testfull.log" ]; then
  printf '%s\n' 'Existing attempt: inspect logs and outputs before explicit recovery.'
  exit 32
fi
"$B/envs/mindbridge/bin/python" -u "$B/mindbridge_reconstruct_k1.py" --pool testfull --generate > "$B/logs/mindbridge-testfull.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$B/logs/mindbridge-testfull.exit"
[ "$rc" -eq 0 ] || exit "$rc"
runtime/env/bin/python -u "$B/evaluate_k1.py" --method mindbridge --ids "$B/test_full_ids.txt" --feature-batch 16 > "$B/logs/mindbridge-score-testfull.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$B/logs/mindbridge-score-testfull.exit"
exit "$rc"
