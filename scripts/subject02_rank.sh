#!/usr/bin/env bash
set -euo pipefail
cd /data1/matengyu/geyugong/neuroadapter-author-first
exec runtime/env/bin/python -B scripts/subject02_train.py "$@"
