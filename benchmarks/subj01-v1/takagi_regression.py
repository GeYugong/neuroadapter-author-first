"""Original Takagi RidgeCV with workspace I/O and persisted fitted pipelines."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import himalaya

B = Path(__file__).resolve().parent
F = B / 'processed/takagi-features'
BR = B / 'processed/regression-brain'
O = B / 'processed/takagi-regression'
SRC = B / 'upstream/takagi/codes/utils/ridge.py'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    assert json.loads((F / 'COMPLETE.json').read_text())['n_ids'] == 9841
    O.mkdir(exist_ok=True)
    for target, width in [('init_latent', 6400), ('c', 59136)]:
        for kind, idsfile in [('each', 'train_trial_ids.npy'), ('ave', 'test_unique_ids.npy')]:
            ids = np.load(BR / idsfile)
            split = 'tr' if kind == 'each' else 'te'
            assert len(ids) == (24980 if split == 'tr' else 982)
            path = O / f'subj01_{kind}_{target}_{split}.npy'
            marker = path.with_suffix('.done')
            if marker.exists():
                assert np.load(path, mmap_mode='r').shape == (len(ids), width)
                continue
            assert not path.exists(), 'Partial array retained; inspect before retry: ' + str(path)
            arr = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(ids), width))
            for row, image_id in enumerate(ids):
                feat = np.load(F / target / f'{image_id:06}.npy').reshape(-1)
                assert feat.size == width and np.isfinite(feat).all()
                arr[row] = feat
            arr.flush()
            del arr
            marker.write_text(now() + '\n')
            print(json.dumps({'event': 'assembled', 'file': str(path), 'shape': [len(ids), width]}), flush=True)
    plan = {'utc': now(), 'source_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(),
            'himalaya_version': himalaya.__version__, 'source_requirement': 'himalaya (unpinned upstream)',
            'brain_train': '24980 single trials, raw original int16 cast float32 by author',
            'brain_test': '982 original repeat means cast float32 by author',
            'feature_order': 'train_trial_ids/test_unique_ids from already verified native brain preparation',
            'IO_adaptation': 'Load per-ID author features into flattened memory-mapped rows; identical flatten order to make_subjstim.py stack then ridge.py reshape. Paths only changed in original ridge.py; fitted pipeline saved additionally.',
            'fit': 'Original StandardScaler and himalaya RidgeCV, original seven alphas, original default CV; early->init_latent and ventral->c sequentially',
            'test_use': 'Original final correlation print only, not parameter or model selection'}
    (B / 'takagi-regression-plan.json').write_text(json.dumps(plan, indent=2) + '\n')


def fit(target):
    roi = 'early' if target == 'init_latent' else 'ventral'
    assert (O / f'subj01_each_{target}_tr.done').exists()
    assert not (O / f'{target}.joblib').exists(), 'Existing fit must not be overwritten'
    tree = ast.parse(SRC.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    rewritten = []
    for node in main.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('mridir', 'featdir', 'savedir'):
                node.value = ast.Constant(str(BR if name == 'mridir' else O) + '/')
        rewritten.append(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'pipeline.fit':
            rewritten.extend(ast.parse(f"joblib.dump(pipeline, {str(O / (target + '.joblib'))!r})").body)
    main.body = rewritten
    ast.fix_missing_locations(tree)
    sys.argv = [str(SRC), '--subject', 'subj01', '--roi', roi, '--target', target]
    started = now()
    exec(compile(tree, str(SRC), 'exec'), {'__name__': '__main__', 'joblib': joblib})
    output = O / f'subj01_{roi}_scores_{target}.npy'
    predicted = np.load(output, mmap_mode='r')
    assert predicted.shape == (982, 6400 if target == 'init_latent' else 59136)
    assert np.isfinite(predicted).all()
    receipt = {'start_utc': started, 'complete_utc': now(), 'target': target, 'roi': roi,
               'shape': list(predicted.shape), 'finite': True, 'original_script_completed': True,
               'source_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(), 'pipeline': str(O / (target + '.joblib'))}
    (O / f'{target}-complete.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU regression only'
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--target', choices=['init_latent', 'c'])
    args = ap.parse_args()
    if args.prepare:
        prepare()
    elif args.target:
        fit(args.target)
    else:
        ap.error('Specify --prepare or --target')
