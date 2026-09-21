# Commands (run on the two RTX 4090 host)

Working directory: `/data1/matengyu/geyugong/neuroadapter-author-first`.
All Python commands use `runtime/env/bin/python`.

```bash
# Full authorized sequential stage; refuses an existing formal run.
tmux new-session -d -s author-stage1 'runtime/env/bin/python scripts/stage1.py > logs/stage1.log 2>&1'
# Manual same-configuration resume to a cumulative target, only when no stage process is running.
runtime/env/bin/python scripts/run_train.py --target 10 --resume runs/subject01-author-fp32-seed42/checkpoint-5
# Independent generation and evaluation (refuses to overwrite existing generated files).
runtime/env/bin/python scripts/evaluate_stage.py --run runs/subject01-author-fp32-seed42 --epoch 5
# Metrics-only recovery after generation completed.
runtime/env/bin/python scripts/evaluate_stage.py --run runs/subject01-author-fp32-seed42 --epoch 5 --metrics-only
```

No cumulative target above 20 is accepted. On errors, inspect logs and complete checkpoints; do not start another trajectory. Use SIGTERM on both training ranks to request epoch-boundary save/exit; an interrupted partial epoch must be replayed from the last COMPLETE checkpoint. Only a new complete readable rolling checkpoint permits cleanup; epochs 1/5/10/20 are always retained.
