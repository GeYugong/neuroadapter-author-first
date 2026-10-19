"""Publish the explicitly authorized Subject 2 inference-weight allowlist."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
os.environ['NEUROADAPTER_HF_RELEASE_DIR'] = str(HERE)
source = HERE.parent / 'neuroadapter-author-first/scripts/publish_hf_release_20261017.py'
spec = importlib.util.spec_from_file_location('publisher', source)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def cards(package, manifest):
    assert package['repo'] == 'gugabobo/neuroadapter-subject02-e109'
    folder = HERE / 'cards'
    folder.mkdir(exist_ok=True)
    public = dict(method=package['method'], source_code_commit=manifest['server_head'],
                  selection=package['selection'],
                  files=[{k: v for k, v in f.items() if k != 'source'} for f in package['files']],
                  scope='Self-trained inference parameters and selected parcel vertex indices only; no raw NSD data.')
    (folder/'manifest.json').write_text(json.dumps(public, indent=2)+'\n', encoding='utf-8')
    (folder/'SHA256SUMS').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in package['files']), encoding='utf-8')
    (folder/'inference-config.json').write_text(json.dumps(package['config'], indent=2)+'\n', encoding='utf-8')
    (folder/'evaluation-summary.json').write_text(json.dumps(dict(n_test=982, metrics=package['metrics'],
        identification_units='percent', other_metric_units='unitless', K1='candidate0 from the same eight',
        best8='Highest correlation between measured fMRI and predicted responses from the official Subject2 16-member encoder; no GT-image selection'), indent=2)+'\n', encoding='utf-8')
    (folder/'README.md').write_text('''---
tags:
- fmri
- nsd
- reproduction
- subject-2
---

# NeuroAdapter Subject 2 E109 reproduction weights

This repository preserves our independently trained Subject 2 adapter used in the completed 982-image evaluation. It is not an official author checkpoint. `inference.pt` contains `guidance_generator`, `image_proj` and `ip_adapter`; `parcel_vertices.pt` fixes the selected 100 parcels per hemisphere. Obtain frozen Stable Diffusion 1.5 and the matching NeuroAdapter architecture separately.

Training used seed42, BF16, two RTX4090 GPUs, effective batch32 (8 per GPU, accumulation2), AdamW LR1e-4, weight decay1e-6, clip1, original Min-SNR gamma5 and TokenDropout. Training ended intentionally after372 epochs; this release is epoch109. Full continuation checkpoints109/300/372 remain preserved separately.

## Checkpoint selection and limitations

On the fixed500-image validation set, epoch372 was better on seven of the eight reconstruction metrics; epoch109 had the highest SSIM. The predeclared balanced gate required SSIM not to decrease, so it conservatively retained109. This does not establish109 as the best overall model. No later candidate passed the K2 gate, so K4 confirmation was not triggered. The released checkpoint was fixed before its final test and was not changed using test scores.

The final982-image test used50 diffusion steps, CFG4, FP32 inference with the original VAE autocast and TF32, candidate batch1 and eight fixed ID/candidate hash seeds. Candidate0 is the K1 control. Best-of-eight uses the official Subject2 sixteen-member brain encoder and measured/predicted fMRI correlation, without GT-image selection. Seven mean metrics improve over K1, while SSIM decreases; no statistical significance is claimed. See `evaluation-summary.json`.

Limitations include a single training seed, validation reused during development, an unknown encoder development split, and different hardware/software/random sequences from the authors. The standard982 IDs were independently derived from the first37 Subject2 sessions; brain inputs retain the registered full40-session three-presentation averages. This is not an independent blind test or an exact reproduction of historical paper execution.

## Provenance and integrity

`manifest.json`, `inference-config.json` and `SHA256SUMS` describe the exact artifact. No raw fMRI, stimulus images, per-image features, credentials, optimizer state, or downloaded backbone weights are included. Upstream architectures and dependencies retain their original licenses.

Upstream: https://github.com/kriegeskorte-lab/NeuroAdapter
Paper: https://arxiv.org/abs/2509.23566
''', encoding='utf-8')
    return folder


publisher.small_files = cards
if __name__ == '__main__':
    try:
        publisher.main()
    except Exception as error:
        publisher.event(event='upload_failed', error_type=type(error).__name__,
            frames=[dict(file=Path(f.filename).name, line=f.lineno) for f in traceback.extract_tb(error.__traceback__)[-4:]])
        sys.exit(1)
