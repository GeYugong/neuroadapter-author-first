# NSD Subject1 benchmark v1 — frozen scope

Authorized 2026-09-26. First actual delivery: MindEye1 full high+low pathways and fixed NeuroAdapter L120, common standard-test T100, K=1 without selection, frozen eight metrics and 32-image comparison atlas. Continue to locked full T immediately; do not wait for other baselines.

Baseline project commit: 8ff1b25b29a641d58e11d92f10463c24980a08fd. NeuroAdapter export SHA256 bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798; epoch120/63840. No new NeuroAdapter training, seed or ROI work. Preserve all historical conclusions and best.

T is the verified intersection of official MindEye1 Subject1 test global NSD IDs and NeuroAdapter untrained standard-test IDs, cross-checked with NSD design. Never substitute internal500. T100: sort T by SHA256('nsd-benchmark-v1:'+str(id)), take100; first32 atlas. Five first IDs for identity/FOV checks. Lock manifests before inference; no test-based model or parameter selection. History includes prior viewing of some test results; this is not globally unseen blind testing.

Native preprocessing and full pathways retained: MindEye15724 nsdgeneral voxels, official high checkpoint last.pth and low epoch120.pth, Versatile Diffusion20 steps/img2img0.85/K1. NeuroAdapter native parcel inputs50 steps/CFG4 and existing42:image_id:0 randomness. Preserve per-model batch-independent random streams. Ground truth cannot condition generation.

One metric implementation: metric_brain_adapter.py (hash recorded separately), EfficientNet-B1 per actual code. Same original NSD GT and full evaluation pool; do not average partitioned identification rates. Separate K1, native selected pipelines, author_outputs_rescored and literature tables. Missing metrics NA with reasons. No claims of equal training budget or independent test tuning.

Parallel Cortex2Image official assets and Brain-Diffuser preparation. Then author-code Brain-Diffuser/Takagi regression modules (frozen generators). MindBridge single-subject600epochs requires measured budget after first delivery; DREAM full brain pathways must be verified; MindFormer missing assets remains blocked, no guessed implementation or unsolicited emails.

All new assets/environments under this workspace; old project read-only. GPU0 MindEye1, GPU1 NeuroAdapter when available; no other tasks killed. Exact upstream commits/download revisions/hashes and exit records retained. Only engineering adapters, no scientific replacements. Report completed methods immediately. Scope follows supplied CODEX_NSD_SUBJECT1_BASELINE_BENCHMARK_ZH.md.

## Parallel MindBridge preparation (2026-09-27)

User authorized preparation alongside Takagi. See [MINDBRIDGE_PREPARATION.md](MINDBRIDGE_PREPARATION.md): official Subject1 data downloads and isolated environment installation are CPU/network only. Original two-GPU600epoch training waits for both GPUs to be available and first-epoch budget measurement; no automatic single-GPU recipe substitution. No new NeuroAdapter work.

## Live transition 2026-09-27 13:37 +0800

Takagi original SD1.4 extraction finished 9841/9841 with exit0; see takagi-feature-complete.json. CPU-only original RidgeCV assembly/early-latent/ventral-caption stages are now sequentially running in bench-takagi-regression, with unchanged author alphas and StandardScaler. himalaya0.4.11 is installed only in the P1 environment (upstream requirement is unpinned). No Takagi reconstruction scores yet.

MindBridge original two-GPU Subject1 Vanilla600epoch training has actually started in bench-mindbridge-training. Its first complete checkpoint records86 optimizer steps,8600 global samples including distributed batching,0 skipped steps,2 ranks. Epoch1 (second actual epoch) is running. Original seed42/batch50 per rank/FP32/DeepSpeedZeRO2/clipping1/OneCycleLR remain. First attempt exited before any rank/update because a newer shared Accelerate default config had an unsupported debug key; preserved attempt1 logs and fixed only explicit workspace config selection. See mindbridge-first-epoch.json and mindbridge-training-start.json. No source training loop changes.

Takagi CPU original early/init_latent and ventral/c RidgeCV both finished exit0 on 2026-09-27; all982 predicted feature rows are finite. Pipelines and prediction hashes are in takagi-regression-complete.json. Image reconstruction and eight-metric evaluation remain pending; no feature correlation is presented as a reconstruction metric. MindBridge currently occupies both GPUs.

## MindBridge checkpoint recovery 2026-09-28

Original run exited1 after an epoch420 distributed conditional-best-save mismatch (421 completed epochs /36206 updates). Full last state was backed up and audited on both ranks. The guarded resume is now bench-mindbridge-resume, logs/mindbridge-resume.log, continuing zero-based421 toward600. Upstream source remains pinned; a bounded runtime patch broadcasts rank0 best decisions, corrects next-epoch resume and persists loader/best metadata. See MINDBRIDGE_RECOVERY_ZH.md and mindbridge-resume-audit-rank0/1.json. Do not restart the old controller or repeat the fixed snapshot launcher. Original OneCycle scheduler had already reached51600; its saved LR is retained, not reset. Takagi features and both original RidgeCV are complete; image generation waits for GPU release.

## Takagi decoder CPU preparation 2026-09-28

MindBridge has passed the next real synchronized checkpoint (zero-based430) and remains training. The Takagi K1 entry takagi_reconstruct_k1.py now compiles the pinned original CVPR decoder body, removes only GT saving, redirects brain-predicted features/output paths and sets native first candidate. CPU preflight verifies test100/full982 ID mapping, feature and SD1.4 SHA; it does not claim GPU inference. Cold model initialization seeds42, cached images restore the exact post-load RNG, and the first fixed target must pass a cold-versus-cached pixel identity check before outputs are admitted. No standard GT is opened by generation. Explicit launch-takagi-test100.sh and launch-takagi-testfull.sh chain generation and unchanged full-pool metric scoring; do not start while the selected GPU has compute processes. First100 report and all32 atlas review must be published before launching full982. Takagi image metrics remain NA until actual generation, scoring and review.

## MindBridge inference CPU preparation 2026-09-28

Original prepare_VD loaded fully on CPU in the isolated MindBridge environment (16 dual transformers, UniPC,FP32,mix0.5), and all982 native test brain repeats/ID mappings were prepared and validated without extracting GT JPG. See mindbridge-inference-cpu-ready.json and MINDBRIDGE_INFERENCE_PROTOCOL_ZH.md. Preserve max-pool EACH repeat before mean; do not pool a pre-averaged brain vector. The future K1 row freezes full600 last.pth, original inference.sh CFG5/default20steps/seed42/mix0.5, with both predicted image/text pathways and no optional AE added. IMPORTANT: original utils.reconstruction uses GT CLIP for native best-of8 selection. K1 adapter must omit that selector/plotting and not accept GT; the GT-dependent native-selected row remains NA under the user prohibition, no invented replacement selector. GPU generation remains pending, do not interrupt or duplicate training.


## MindBridge K1 generation-prefix adapter preparation

mindbridge_k1_native.py now compiles the pinned original generation prefix into a GT-free function, statically preserving the frozen Vanilla branch. test_mindbridge_k1_native.py passed CPU synthetic-module differential tests against the original prefix, including both predicted conditions, local RNG and forbidden CLIP selector checks; see mindbridge-k1-adapter-cpu-test.json. No real model or GPU candidate was generated. The full guarded checkpoint/ID/output runner remains to be integrated after reading these reusable preparations; actual600 completion and GPU prefix identity are still required before benchmark outputs.


## MindBridge guarded runner CPU ready

mindbridge_reconstruct_k1.py and launch-mindbridge-test100.sh/testfull.sh now integrate the prepared original generation function and strict MindSingle load. Both CPU ID/brain/SHA preflights pass; live-training rejection was exercised without creating outputs. Generation requires a real independently checked mindbridge-training-complete.json (600epochs/51600updates/exit0/full_state_verified/checkpoint_sha256), final599 last metadata and a free selected GPU. No completion receipt exists while training is live. Actual first-target native-prefix versus GT-free GPU tensor identity is mandatory; one additional check, no candidate selection. Evaluate original FP32 tensor, retain PNG only for atlas. Full982 additionally requires actual first100 scores/32atlas review/publication receipt; see protocol for schema. No actual GPU reconstruction has run yet.


## Live transition 2026-09-28 07:44 +0800

MindBridge full600 exited0; final export equals both ZeRO partitions,51600 updates and protected inputs unchanged. See mindbridge-training-complete.json. GPU identity initially failed due native repeated-forward nondeterminism; failed attempt retained. Process-local deterministic algorithms plus CUBLAS workspace yielded exact original-prefix/adapter FP32 equality; numerical-runtime deviation explicitly disclosed, no science parameter changes. Retry is now generating fixed100 onGPU1. Takagi test100 generation/scoring exit0; all32 atlas AI reviewed and texture/semantic failures disclosed in TAKAGI_test100_ids_ZH.md. Publish this real result then continue locked full982, retaining existing100 and settings.

## Live transition 2026-09-28 07:51 +0800

Takagi test100 published d83e69c; full982 is running in bench-takagi-testfull onGPU0, reusing existing100. MindBridge600 last actual test100 generation/scoring both exit0; all32 atlas AI reviewed with failures retained. Publish MINDBRIDGE_test100_ids_ZH.md and review, then create a real publication receipt with that commit and launch existing launch-mindbridge-testfull.sh onGPU1. No retraining or scientific parameter changes.

MindBridge test100 published fc60df6 (GitHub private repository and all4 atlas files verified). Actual publication receipt written; existing launch-mindbridge-testfull.sh started in bench-mindbridge-testfull onGPU1. It reuses verified100 FP32 tensors and generates remaining882, then scores the entire982 negative pool. Takagi full982 remains onGPU0. Do not duplicate either controller.

## Full K1 completion 2026-09-28

Takagi and MindBridge982 generation/scoring exit0; all PNG and MindBridge original FP32 output SHA verified; fixed32 atlas byte-identical; protected files unchanged; GPUs released. Publish independent full reports, unified100/982 CSV and completed model manifest. Main K1 comparison covers5 real methods with3 explicit asset blocks. Native multi-candidate B table remains a separate follow-up: MindEye pinned selector uses brain-predicted proj_embeddings, not GT. Audit exact16-candidate requirements and resource budget before execution; never mix with K1 or modify existing outputs. MindBridge native GT selector disallowed. No new NeuroAdapter training.
