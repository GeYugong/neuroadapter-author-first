"""Prepare an explicit public weight manifest; never upload raw NSD data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data1/matengyu/geyugong/neuroadapter-author-first')
OUT = ROOT / 'reports/hf-release-20261017'
OUT.mkdir(parents=True, exist_ok=True)
B = ROOT / 'benchmarks/subj01-v1'


def emit(x):
    print(json.dumps(x, ensure_ascii=False), flush=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def row(path, destination):
    path = path.resolve()
    path.relative_to(ROOT)
    assert path.is_file()
    value = dict(source=str(path.relative_to(ROOT)), path=destination,
                 size=path.stat().st_size, sha256=sha(path))
    emit(dict(event='hashed', **value))
    return value


import numpy as np
# Store only aggregate training-feature statistics required by the author's
# post-regression calibration. No image/brain sample arrays enter this release.
for kind in ('vision', 'text'):
    target = OUT / f'clip{kind}-target-normalization.npz'
    if not target.exists():
        emit(dict(event='derive_target_statistics', kind=kind))
        data = np.load(B / f'processed/brain-diffuser-features/nsd_clip{kind}_train.npy')
        means = np.stack([np.mean(data[:, i], axis=0) for i in range(data.shape[1])])
        stds = np.stack([np.std(data[:, i], axis=0, ddof=0) for i in range(data.shape[1])])
        assert np.isfinite(means).all() and np.isfinite(stds).all()
        np.savez(target, train_target_mean=means, train_target_std_ddof0=stds,
                 n_train=np.array(data.shape[0]))
        del data

packages = []
base = ROOT / 'runs/subject01-author-fp32-seed42-lr1e-5-from400/exports/epoch-450'
packages.append(dict(repo='gugabobo/neuroadapter-subject01-l450', method='NeuroAdapter L450',
    selection='Selected on development validation; later single-variable branches did not replace L450.',
    files=[row(base/n, n) for n in ('inference.pt', 'parcel_vertices.pt')],
    metrics={
      'k1': json.loads((ROOT/'reports/balanced-lr-from400/test982/metrics/metrics.json').read_text()),
      'brain_encoder_best8': json.loads((ROOT/'reports/l450-best8-test982/comparison.json').read_text())}))

mb = B/'upstream/mindbridge/train_logs/benchmark-subj01-vanilla-seed42'
packages.append(dict(repo='gugabobo/mindbridge-subject01-reproduction', method='MindBridge single-subject',
    selection='last.pth: fixed 600-epoch endpoint used in reported inference; best.pth: native validation-selected checkpoint, not separately test-selected.',
    files=[row(mb/n,n) for n in ('last.pth','best.pth')],
    metrics={'native8_gt_clip':json.loads((B/'metrics/mindbridge-native8/test_full_ids/metrics.json').read_text())}))

tk=B/'processed/takagi-regression'
packages.append(dict(repo='gugabobo/takagi-subject01-regression', method='Takagi 2023 regression',
    selection='Original RidgeCV fitted pipelines for early visual ROI -> initial latent and ventral ROI -> text conditioning; no checkpoint sweep.',
    files=[row(tk/n,n) for n in ('init_latent.joblib','c.joblib')], metrics={}))

bd=B/'processed/brain-diffuser-regression'
files=[]
for kind in ('vdvae','clipvision','cliptext'):
    files.append(row(bd/kind/f'{kind}_regression_weights.pkl',f'{kind}/regression_weights.pkl'))
    norm='normalization.npz' if kind=='vdvae' else 'brain-normalization.npz'
    files.append(row(bd/kind/norm,f'{kind}/{norm}'))
    if kind!='vdvae':
        files.append(row(OUT/f'{kind}-target-normalization.npz',f'{kind}/target-normalization.npz'))
packages.append(dict(repo='gugabobo/brain-diffuser-subject01-regression',method='Brain-Diffuser regression',
    selection='Author fixed-alpha regressions; VDVAE alpha50000, CLIP vision60000, CLIP text100000. Not selected on test scores.',
    files=files,metrics={'single_candidate':json.loads((B/'metrics/brain-diffuser/test_full_ids/metrics.json').read_text())}))

manifest=dict(created_unix=time.time(), server_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    packages=packages, excluded=['Senior E750: user explicitly excluded', 'MindEye1: downloaded official pretrained weights',
    'Frozen SD/VD/CLIP/VDVAE and official brain encoders: downloaded pretrained weights',
    'Raw NSD images, brain responses, per-sample training features, credentials'],
    notes=['Inference weights, not full optimizer/RNG backups. Full final resume states remain on the server.',
    'Existing historical L120 and selection-4090 Hub repositories are preserved.'])
(OUT/'upload-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
emit(dict(event='manifest_ready',packages=len(packages),bytes=sum(f['size'] for p in packages for f in p['files'])))
