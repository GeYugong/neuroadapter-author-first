#!/usr/bin/env bash
set -u
set -o noclobber
cd /data1/matengyu/geyugong/neuroadapter-author-first || exit 1
if [[ -e logs/lr-half-to500-controller.log || -e logs/lr-half-to500-controller.exit ]]; then
  echo 'Prior controller logs exist; inspect before explicit recovery.' >&2
  exit 1
fi
runtime/env/bin/python -u scripts/lr_half_to500.py --resume /data1/matengyu/geyugong/neuroadapter-author-first/runs/subject01-author-fp32-seed42-lr5e-5-from100/checkpoint-300 --launch > logs/lr-half-to500-controller.log 2>&1
rc=$?
printf '%s\n' "$rc" > logs/lr-half-to500-controller.exit
exit "$rc"
