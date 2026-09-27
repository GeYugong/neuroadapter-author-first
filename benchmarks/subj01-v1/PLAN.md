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
