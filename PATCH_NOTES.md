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

Stage3: explicit --stage3 full resume target60; default stage1 and stage2 limits retained. Separate stage3 controller reuses unchanged evaluation path (val100 only). Protect60 and existing49 in addition to all previous milestones; best summary includes60. Only checkpoint cleanup and argument parser function ASTs changed, no scientific functions changed.

Stage4: explicit full-resume targets70/80 retain stage1/2/3 boundaries. Small controller reuses stage3 execution/summary helpers and existing trainer/evaluator; only80 added to full500 evaluation path. Protect70/80 and existing59; best candidates extend through80. Trainer scientific functions unchanged; only cleanup/parser function ASTs changed.

Stage5: explicit full-resume90/100 targets, preserving all earlier stage boundaries. Small stage controller reuses stage3 execution/summary; evaluator adds100 to existing full500 path. Protect90/100 and existing79. Only trainer cleanup/parser function ASTs changed; no scientific functions changed.

Stage6: explicit full-resume110/120, prior stage boundaries preserved. Reuse stage3 execution/summary and existing trainer/metrics. Add120 to full500; protect110/120 and existing99. Only trainer cleanup/parser ASTs changed; scientific functions unchanged.


## Fixed L120 IBBI and ROI pilot (2026-09-24)

- Added isolated orchestration and read-only hooks around the unchanged author decoder, attention processor, and evaluator. No trainer, optimizer, model, loss, sampling or existing metric implementation is changed.
- Official Subject1 native functional labels are mapped by the NSD hemisphere-specific white-to-fsaverage nearest-neighbor transforms, with one-based stored source indices converted to zero-based. Saved L120 parcel order/vertices and the SNR fsaverage index are checked exactly.
- Explicit strict per-raw-label overlap >0.5 replaces neither historical results nor the author's plotting default0.75/dataset >= comparison; this experiment uses31 LL and78 HL parcels and preserves the two cross-memberships.
- Original pure attention integration/ROI-map/overlay functions are loaded from their AST definitions to avoid the plotting entry's unrelated hardcoded cwd, pycortex setup and dataset defaults. Streaming adapts storage only: query-weighted B, layer-L1/equal-layer RAM.
- A conditional E clone is masked after GuidanceGenerator and before original ImageProjModel, retaining every unselected row and the original unconditional path. Bias/LayerNorm remain. This is an explicit model-internal intervention, not a verified exact Table14 implementation.
- Four original decode2 candidates, with both read-only attention and an explicit empty mask, matched prior L120 outputs pixel-exact; those four count within the fixed64 IBBI trajectories.
- Three complement-only random sets per ROI use seed2026092403 and exact parcel/hemisphere counts; SNR, vertex size, function and mesh adjacency are reported but not matched.
