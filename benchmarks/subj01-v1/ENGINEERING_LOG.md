# Engineering record

- 2026-09-26: source root and GPUs checked, no competing benchmark;454GiB free. Original environments/history unchanged.
- Isolated Python3.10 environment created under envs/mindeye1. OS Python lacked ensurepip; pip was bootstrapped into only this venv via existing pip --python. No system/shared package changes.
- Pinned author core versions. Restored original compatible setuptools/pydantic/torchmetrics versions after dependency resolution; final freeze will be captured.
- Official low epoch120 contains maps_projector training-head weights. Author default use_cont=False plus strict=False ignores these. We instantiate original use_cont=True and strictly load every key; default forward does not invoke projector. CPU zero-input equality with projector replaced by Identity verified exactly. No low-path fallback.
- Unused clip-retrieval package import is disabled only in an in-memory utils module; any attempted retrieval raises. Original reconstruction is unchanged, K1 returns candidate0. Local CLIP assets are explicit and pinned.
- Downloads use anonymous official pinned revisions; existing partial bytes retained. Multi-range retries preserve partial chunks, final SHA must match official LFS. Only this task download workers were restarted, no GPU processes touched.
- Cortex2Image official Box weight/reconstruction and surface-data links both404; no Subject1 files or IDs acquired; no inferred result.

2026-09-26: tmux inherited lowercase proxy variables pointing at stale port17897, overriding newly provided uppercase values. Download-only adapters now pass explicit proxies at19876; verified HTTP206 exact8MiB range. Existing verified segments and partial prefixes retained. No inference protocol changes.

Five-ID identity check: predetermined NSD18483 zebra pair is visually the same image/FOV, but raw425 vs JPEG256-upsampled pixel correlation is0.9072 due detail/resampling. Removed arbitrary0.98 automatic threshold; keep all five correlations and require actual visual identity/FOV review. No ID, GT, evaluation resize or scientific setting changed.

MindEye high strict loading rejected one extra official checkpoint parameter voxel2clip.temp (author models.py line177 comments out its registration; forward does not reference it). Retain/register the scalar training temperature parameter to load all keys strictly, rather than silently ignore unexpected state. First attempt produced zero images and exited1; its protocol and logs preserved. Retry uses unchanged author forward/prior/reconstruction.
