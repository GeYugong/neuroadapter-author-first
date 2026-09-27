#!/bin/bash
# Explicit launch only after checking MindBridge has exited and this GPU is free.
set -u
cd /data1/matengyu/geyugong/neuroadapter-author-first || exit 1
B=benchmarks/subj01-v1
export CUDA_VISIBLE_DEVICES="${1:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ -e "$B/logs/takagi-testfull.log" ] || [ -e "$B/logs/takagi-score-testfull.log" ]; then
  printf '%s\n' 'Existing attempt: inspect stack and outputs; never overwrite or blindly retry.'
  exit 32
fi
"$B/envs/p1-author310/bin/python" -u "$B/takagi_reconstruct_k1.py" --pool testfull --generate > "$B/logs/takagi-testfull.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$B/logs/takagi-testfull.exit"
[ "$rc" -eq 0 ] || exit "$rc"
runtime/env/bin/python -u "$B/evaluate_k1.py" --method takagi --ids "$B/test_full_ids.txt" --feature-batch 16 > "$B/logs/takagi-score-testfull.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$B/logs/takagi-score-testfull.exit"
exit "$rc"
