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
