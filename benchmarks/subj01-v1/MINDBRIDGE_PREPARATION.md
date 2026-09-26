# MindBridge Subject1 Vanilla — parallel preparation

User authorized parallel preparation on 2026-09-27 while Takagi CVPR feature extraction continues. This is a baseline, not a new NeuroAdapter experiment.

- Official code: https://github.com/littlepure2333/MindBridge, commit `2a3943ff5b619db6f85f16a5e7ff1331d25def55`.
- Variant: Subject1-only `MindSingle` / paper Vanilla. Do not use public `subj1257` weights.
- Recipe: original `scripts/train_single.sh`, two GPUs, 600 epochs, per-rank batch50, validation batch50, hidden2048, 4 blocks, max pooling8192, MSE multiplier10000, seed42, original augmentation, AdamW and OneCycleLR. CLI maximum LR is 3e-4; upstream main multiplies it by world size (two-rank argument becomes6e-4 internally). Do not silently replace this with the author's single-GPU variant.
- Budget: first actual original-config epoch will measure time, memory, update count and a 600-epoch estimate. No shortened run may be labeled complete. No benchmark-test-based checkpoint or parameter selection.

## Current preparation and scheduling

`bench-mindbridge-data` runs `launch-mindbridge-prepare.sh`; `bench-mindbridge-env` runs `launch-mindbridge-install.sh`. Both have CUDA_VISIBLE_DEVICES empty. They only use CPU/network/storage. Training has not started, and neither launcher starts training automatically.

Takagi keeps GPU1. Do not kill, pause, or contend with it. Original two-rank MindBridge training waits for verified availability of both GPUs and completed data/environment checks. Other users' processes must not be terminated. No fabricated wall-time estimate before throughput measurement.

## Data lock

`mindbridge-data-lock.json` freezes official `pscotti/naturalscenesdataset` revision `421fb80ea4ca5600c7364ed400f9a5941380c387`, 18 Subject1 train shards and one validation shard, 38,670,315,520 bytes, individual LFS SHA256 hashes. Test is not added to training.

Download reuses existing SHA-verified resumable transport in `download_p1_models.py`. Unpack only original JPEG, nsdgeneral.npy and coco73k.npy files required by the author loader, without image/brain conversion. Expected8559 train and300 validation samples; actual counts, ID uniqueness, original3x15724 brain shape/finiteness and disjointness from locked982 standard-test IDs must pass. Original NSDAccess COCO2017 captions and stimulus metadata are linked read-only. Full completion marker means preparation only, not training.

## Environment

Isolated `envs/mindbridge`, Python3.10.12. Retain author's pinned torch2.0.1, torchvision0.15.2 and diffusers0.13.0; additional compatibility pins are in `mindbridge-requirements.txt`. Author's unversioned `clip` requirement is supplied by the OpenAI-API-compatible clip-anytorch package already used in this benchmark. Record full installed versions and validate original-module CPU imports. A completed installation is not a GPU/DeepSpeed training validation. No existing NeuroAdapter, MindEye or P1 environment is modified.

## Recovery / exit evidence

Logs: `logs/mindbridge-data-prepare.log`, `logs/mindbridge-env-install.log`, and corresponding `.exit` files. Preserve failed-attempt logs; fix only engineering transport/dependency issues. Download can resume `.part` files, checks full source SHA, and refuses mismatching existing extracted files. Do not remove completed or historical outputs. Watch actual state before retrying to prevent duplicate processes.

Before training: original modules and frozen CLIP assets must load, training/validation IDs must pass the locked test exclusion check, full-size model must fit, and source/settings must be recorded. Final inference follows the benchmark's K1 and common test protocol; original multi-candidate selection remains a separate table. All existing NeuroAdapter weights, best and prior results remain unchanged.
