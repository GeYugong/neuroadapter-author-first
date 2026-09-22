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

## Stage 4: cumulative70/80 only

```bash
runtime/env/bin/python scripts/stage4.py --target 70
# Only after AI review and publication of70, and checking GPU/process availability:
runtime/env/bin/python scripts/stage4.py --target 80
# Recovery after failure only from the latest COMPLETE epoch after all prior processes exit:
runtime/env/bin/python scripts/run_train.py --stage4 --target 70 --resume runs/subject01-author-fp32-seed42/checkpoint-60
```

Stage4 explicitly allows70/80; old stage limits remain.70 uses monitor100;80 generates500 once and scores500 and monitor100 from the same outputs. Science unchanged. Every invocation ends after evaluation; review/publish70 before starting80. Full60 and all historical milestones remain protected;70/80 are permanent.80 completes and stops without automatic further training.

## Epoch70 full500 supplement (evaluation only)

`runtime/env/bin/python scripts/complete_epoch70_val500.py` verifies original epoch70 export (37240 updates), reuses unchanged monitor100 outputs, and generates only400 missing IDs with the original decoder (200 disjoint IDs on each GPU; batch1, two sequential candidates per image). New files live in `runs/subject01-author-fp32-seed42/decoded/epoch-70-val500-supplement`; a source manifest and symlink union assemble500 by original validation ID order. Existing outputs are not overwritten. The command refuses existing supplement/report directories; inspect failures before any recovery.

Author metrics score each candidate on the complete500 pool separately. The comparison reuses epoch80 CSVs, averages two candidates within each image, then applies10000 paired image bootstrap draws (NumPy PCG64 seed20260922), reporting95% percentile intervals for epoch70 minus epoch80. Identification scores keep the fixed500 negative pool. Intervals are exploratory internal-development-set results, not independent-test or cross-training-seed conclusions. No training or best-selection function is called.
