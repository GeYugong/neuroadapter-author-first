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


## Stage 2: authorized cumulative 50 epochs

Stage 1 remains capped at 20 by default. Stage 2 requires --stage2 and a full checkpoint; allowed cumulative targets are 30/40/50. No fresh initialization is authorized.

```bash
runtime/env/bin/python scripts/stage2.py --target 30
runtime/env/bin/python scripts/stage2.py --target 40
runtime/env/bin/python scripts/stage2.py --target 50
# Recovery only after all previous processes exit and inspecting the last COMPLETE epoch:
runtime/env/bin/python scripts/run_train.py --stage2 --target 30 --resume runs/subject01-author-fp32-seed42/checkpoint-20
```

Each stage invocation exits after evaluation. The monitoring assistant reviews every fixed atlas as AI, publishes that milestone immediately, checks the authorized stop criteria, then starts the next invocation. No hyperparameter approval framework is introduced. Targets refer to cumulative epochs, not additional epochs. Full checkpoints 1/5/10/20/30/40/50 are permanent; other rolling checkpoints retain the existing last-two policy. Non-finite loss/gradients stop before a corrupt optimizer update; the previous COMPLETE epoch remains the recoverable state (no claim of mid-epoch recovery). Full recovery records resume-to-30/40/50.json before training, including exact AdamW-state equality against the saved checkpoint and actual LR.

## Stage 3: cumulative60 only

`runtime/env/bin/python scripts/stage3.py --target 60` starts from full checkpoint50, trains then exits training before independent val100 evaluation and stops. Explicit recovery command only after checking all processes have exited: `runtime/env/bin/python scripts/run_train.py --stage3 --target 60 --resume runs/subject01-author-fp32-seed42/checkpoint-50`. Existing checkpoints including49 remain protected. No full500 evaluation at60.
