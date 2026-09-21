# Minimal patches

- train_brain_adapter.py: parameterized dataset paths; conditional multi-subject import; remove embedded W&B credential; stable run ID. No new scientific method.
- setup_accelerator/setup_dataset_and_dataloader: Accelerate SeedableRandomSampler and isolated DataLoader generator; persist epoch and generator state. Existing sampler, no custom sample planner.
- process_training_batch: flatten tokenizer singleton batch axis for CLIPTextModel; no content or loss change.
- training_loop: clip only at sync boundary, count actual updates and padded samples; full Accelerate save/load with progress, atomic completed directories, retain milestones and last two rolling states.
- dataset.__getitem__: add image_id only, original HDF5 averaging/parcels/padding/image transforms untouched.
- decode.run_diffusion: explicit generator for VAE latent, initial noise and DDPM scheduler noise; algorithm/CFG/VAE/autocast unchanged.
- ip_adapter.py: compatibility fallback for relocated, unused MultiControlNetModel import; ImageProjModel unchanged.

Unchanged scientific functions: load_pretrained_models, setup_ip_adapter, setup_models_and_optimizer, apply_fmri_token_dropout; ParcelMapper, GuidanceGenerator, NeuroAdapter, ImageProjModel, attention processors, min_snr_loss_weights.

Short-resume comparison tolerance declared before execution: identical image IDs and RNG tensors, model/AdamW floating tensors allclose(rtol=1e-6,atol=1e-7), integer states exact. No post-hoc relaxation.

- decode main/create_test_dataset: val/train split and exact image ID entry, strict export loading and vertex identity check, no-selection branch calls original run_diffusion twice with fixed candidate batch 1. Original CUDA VAE decode autocast is retained and explicitly recorded.
- metric_brain_adapter.py: only CLIP and SwAV local asset paths parameterized; all eight metric functions, preprocessing, layer choices and two-way direction unchanged.
- scripts: launch/stage orchestration, author metric calls, export comparisons and fixed contact sheets. No new forward, loss, sampler, or metric formulas.
- Checkpoint directory precondition now synchronized across ranks after first short-test failure.


Stage 2: explicit --stage2 targets 30/40/50 retain the default stage1 <=20 boundary; separate stage2 controller starts from full previous milestone. Evaluation adds epoch50 to full500 path, unchanged decoding/scoring. Permanent checkpoint list extends to 30/40/50; stage2 best selection considers all completed 5/10/20/30/40/50 nodes. Added post-load exact AdamW-state audit and non-finite loss/gradient stop guards, with no loss, model, optimizer, sampling, accumulation or padding formula changes. Stage1 controller and historical results remain unchanged.
