#!/bin/bash
set -uo pipefail
cd /data1/matengyu/geyugong/neuroadapter-author-first
B="$PWD/benchmarks/subj01-v1"
exec 9>"$B/processed/takagi-regression.lock"
flock -n 9 || exit 73
test ! -e "$B/logs/takagi-regression.log" || exit 74
exec >"$B/logs/takagi-regression.log" 2>&1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY="$B/envs/p1-author310/bin/python"
rc=0
"$PY" -u "$B/takagi_regression.py" --prepare || rc=$?
if [ "$rc" -eq 0 ]; then "$PY" -u "$B/takagi_regression.py" --target init_latent || rc=$?; fi
if [ "$rc" -eq 0 ]; then "$PY" -u "$B/takagi_regression.py" --target c || rc=$?; fi
printf '%s\n' "$rc" > "$B/logs/takagi-regression.exit"
exit "$rc"
