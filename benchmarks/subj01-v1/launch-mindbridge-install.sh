#!/bin/bash
set -uo pipefail
cd /data1/matengyu/geyugong/neuroadapter-author-first
exec > benchmarks/subj01-v1/logs/mindbridge-env-install.log 2>&1
export CUDA_VISIBLE_DEVICES=""
# tmux inherited obsolete proxy17897; this task uses direct PyPI transport.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export DS_BUILD_OPS=0
export MAX_JOBS=4
export PIP_CACHE_DIR="$PWD/benchmarks/subj01-v1/cache/mindbridge-pip"
PY="$PWD/benchmarks/subj01-v1/envs/mindbridge/bin/python"
# Old DeepSpeed build needs a preinstalled torch; no shared environment edits.
"$PY" -m pip install --disable-pip-version-check 'setuptools<70' wheel torch==2.0.1 torchvision==0.15.2
rc=$?
if [ "$rc" -eq 0 ]; then
  "$PY" -m pip install --disable-pip-version-check --no-build-isolation -r benchmarks/subj01-v1/mindbridge-requirements.txt
  rc=$?
fi
if [ "$rc" -eq 0 ]; then
  "$PY" -m pip freeze > benchmarks/subj01-v1/mindbridge-environment.txt
  cd benchmarks/subj01-v1/upstream/mindbridge/src
  "$PY" -c 'import torch, torchvision, clip, accelerate, deepspeed, models, trainer, data, nsd_access; assert hasattr(clip, "load"); print("AUTHOR_CPU_IMPORTS_OK", torch.__version__, torchvision.__version__, torch.cuda.device_count())'
  rc=$?
  cd /data1/matengyu/geyugong/neuroadapter-author-first
fi
printf '%s\n' "$rc" > benchmarks/subj01-v1/logs/mindbridge-env-install.exit
exit "$rc"
