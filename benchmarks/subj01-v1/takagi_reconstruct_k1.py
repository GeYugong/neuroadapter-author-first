"""Execute the original CVPR decoder body with predicted features and explicit IDs.

CPU preflight is safe while training runs. GPU generation is separately guarded.
The first image checks cold-load versus cached-model RNG before admitting outputs.
"""
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np

B = Path(__file__).resolve().parent
U = B / 'upstream/takagi'
SD = U / 'codes/diffusion_sd1/stable-diffusion'
SRC = U / 'codes/diffusion_sd1/diffusion_decoding.py'
SOURCE_SHA = '23cfc426025e1a9ccd1c00ba994758d70e685ac9de65a0068340353cffd08d8b'
MODEL_SHA = 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'
MODEL = B / 'assets/p1-models/takagi/sd-v1-4.ckpt'
PRED = B / 'processed/takagi-regression'
OUTPUT = B / 'outputs/takagi'
NATIVE = B / 'outputs/takagi-native'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def savej(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def author_code():
    assert sha(SRC) == SOURCE_SHA
    tree = ast.parse(SRC.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('load_model_from_config', 'load_img_from_arr')]
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    def target(node):
        return node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None
    start = next(i for i, n in enumerate(main.body) if target(n) == 'n_samples')
    body = ast.Module(body=main.body[start:], type_ignores=[])
    counts = dict(K=0, output=0, predicted_z=0, predicted_c=0, remove_GT_save=0)
    class Adapter(ast.NodeTransformer):
        def visit_Assign(self, node):
            name = target(node)
            replacements = {'n_iter': ('K', '1'), 'outdir': ('output', '_NATIVE_OUT'),
                            'scores_latent': ('predicted_z', '_LATENTS'), 'scores_c': ('predicted_c', '_CONDITIONS')}
            if name in replacements:
                key, expr = replacements[name]
                counts[key] += 1
                node.value = ast.parse(expr, mode='eval').body
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if 'sdataset' in ast.unparse(node):
                assert '_org.png' in ast.unparse(node)
                counts['remove_GT_save'] += 1
                return None
            return self.generic_visit(node)
    body = Adapter().visit(body)
    assert all(v == 1 for v in counts.values()), counts
    ast.fix_missing_locations(body)
    assert 'sdataset' not in ast.unparse(body)
    return (compile(ast.Module(body=functions, type_ignores=[]), str(SRC), 'exec'),
            compile(body, str(SRC), 'exec'), counts)


def preflight(pool):
    code = author_code()
    assert subprocess.check_output(['git', '-C', str(U), 'rev-parse', 'HEAD'], text=True).strip() == 'e187d4b3db1d647ee3e1b4256a2068ffd15df683'
    native = np.load(B / 'processed/regression-brain/test_unique_ids.npy').reshape(-1)
    full = np.loadtxt(B / 'test_full_ids.txt', dtype=int).reshape(-1)
    ids_path = B / ('test100_ids.txt' if pool == 'test100' else 'test_full_ids.txt')
    ids = np.loadtxt(ids_path, dtype=int).reshape(-1)
    assert len(native) == len(set(native.tolist())) == 982
    assert set(native.tolist()) == set(full.tolist())
    assert len(ids) == (100 if pool == 'test100' else 982)
    assert set(ids.tolist()) <= set(native.tolist())
    receipt = json.loads((B / 'takagi-regression-complete.json').read_text())
    assert receipt['exit_code'] == 0
    assert sha(MODEL) == MODEL_SHA
    hashes = {}
    for row in receipt['artifacts']:
        if row['path'].endswith('.npy'):
            assert sha(B / row['path']) == row['sha256']
            hashes[row['path']] = row['sha256']
    latent = np.load(PRED / 'subj01_early_scores_init_latent.npy', mmap_mode='r')
    condition = np.load(PRED / 'subj01_ventral_scores_c.npy', mmap_mode='r')
    assert latent.shape == (982, 6400) and condition.shape == (982, 59136)
    assert np.isfinite(latent).all() and np.isfinite(condition).all()
    plan = dict(utc=now(), pool=pool, n_eval=len(ids), ids_sha256=sha(ids_path),
                source_sha256=SOURCE_SHA, adapter_sha256=sha(Path(__file__).resolve()),
                model_sha256=MODEL_SHA, prediction_sha256=hashes,
                variant='Original Takagi CVPR SD1.4; predicted early z and ventral c',
                K=1, selection='none; native first candidate', ddim_steps=50, t_enc=40,
                strength=.8, CFG=5., eta=0., resolution=512, seed=42,
                precision='Original autocast/EMA scopes; original posterior encoding outside autocast',
                gt_access=False, ast_adaptations=code[2],
                randomness='Seed42 before author model construction; cold first image then restore exact post-load Python/NumPy/CPU/CUDA RNG for cached-model images',
                cache_validation='First target cold and cached outputs must be pixel-identical; extra check is not candidate selection',
                gpu_inference_verified=False)
    return code, native, ids, latent, condition, plan


def generate(pool, resume):
    code, native, ids, latent, condition, plan = preflight(pool)
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in ('0', '1'), 'Explicit single physical GPU required.'
    uuid = subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(line.startswith(uuid) for line in jobs.splitlines()), 'Selected GPU is occupied; never evict training.'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    NATIVE.mkdir(parents=True, exist_ok=True)
    with open(NATIVE / 'generation.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = NATIVE / f'{pool}.started.json'
        assert resume or not marker.exists(), 'Previous attempt exists. Inspect logs and use explicit --resume.'
        assert not (B / f'takagi-{pool}-complete.json').exists(), 'Pool already completed.'
        attempt = dict(plan, physical_gpu=physical, pid=os.getpid(), explicit_resume=resume)
        if marker.exists():
            history = json.loads(marker.read_text())
            assert history['adapter_sha256'] == plan['adapter_sha256'], 'Inspect code changes before resuming.'
            attempt['previous_attempts'] = history.get('previous_attempts', []) + [
                {k: v for k, v in history.items() if k != 'previous_attempts'}]
        savej(marker, attempt)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        import PIL
        from PIL import Image
        from omegaconf import OmegaConf
        from einops import rearrange
        from torch import autocast
        from contextlib import nullcontext
        from tqdm import trange
        from pytorch_lightning import seed_everything
        sys.path.insert(0, str(SD))
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler
        ns = dict(torch=torch, np=np, PIL=PIL, Image=Image, os=os, rearrange=rearrange,
                  autocast=autocast, nullcontext=nullcontext, trange=trange,
                  instantiate_from_config=instantiate_from_config, DDIMSampler=DDIMSampler)
        exec(code[0], ns)
        seed_everything(42)
        config = OmegaConf.load(SD / 'configs/stable-diffusion/v1-inference.yaml')
        config.model.params.cond_stage_config.params = {'version': str(B / 'assets/clip-vit-large-patch14')}
        torch.cuda.set_device(0)
        model = ns['load_model_from_config'](config, str(MODEL), 0)
        post_load = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        def run_one(nsd_id, folder, cold=False):
            if not cold:
                random.setstate(post_load[0]); np.random.set_state(post_load[1])
                torch.set_rng_state(post_load[2]); torch.cuda.set_rng_state_all(post_load[3])
            row = int(np.flatnonzero(native == nsd_id)[0])
            scope = dict(ns, model=model, gpu=0, method='cvpr', subject='subj01', imgidx=row,
                         stims_ave=native, tr_idx=np.zeros_like(native), _LATENTS=latent,
                         _CONDITIONS=condition, _NATIVE_OUT=str(folder))
            exec(code[1], scope)
            return folder / 'samples' / f'{row:05}_000.png'
        records = NATIVE / 'records'; records.mkdir(exist_ok=True)
        validation_path = B / 'takagi-decoder-cache-validation.json'
        if not validation_path.exists():
            first = int(np.loadtxt(B / 'test100_ids.txt', dtype=int).reshape(-1)[0])
            cold = run_one(first, NATIVE / 'identity-cold', cold=True)
            cached = run_one(first, NATIVE / 'identity-cached')
            identical = np.array_equal(np.array(Image.open(cold)), np.array(Image.open(cached)))
            savej(validation_path, dict(utc=now(), first_id=first, cold_sha256=sha(cold),
                                       cached_sha256=sha(cached), pixel_identical=identical,
                                       extra_identity_candidates=1, source_sha256=SOURCE_SHA,
                                       adapter_sha256=plan['adapter_sha256']))
            assert identical, 'Cache identity failed; preserve both outputs, do not continue.'
        else:
            validation = json.loads(validation_path.read_text())
            assert validation['pixel_identical'] and validation['source_sha256'] == SOURCE_SHA
            assert validation['adapter_sha256'] == plan['adapter_sha256']
        for value in ids:
            nsd_id = int(value); record = records / f'{nsd_id}.json'; output = OUTPUT / f'{nsd_id}.png'
            if record.exists():
                info = json.loads(record.read_text())
                assert info['nsd_id0'] == nsd_id and info['source_sha256'] == SOURCE_SHA
                assert info['model_sha256'] == MODEL_SHA and sha(output) == info['png_sha256']
                assert info['adapter_sha256'] == plan['adapter_sha256']
                continue
            assert not output.exists(), 'Unaccounted output: inspect, do not overwrite.'
            if nsd_id == int(json.loads(validation_path.read_text())['first_id']):
                row = int(np.flatnonzero(native == nsd_id)[0])
                generated = NATIVE / 'identity-cold/samples' / f'{row:05}_000.png'
            else:
                generated = run_one(nsd_id, NATIVE / 'formal')
            shutil.copy2(generated, output)
            info = dict(nsd_id0=nsd_id, native_test_row=int(np.flatnonzero(native == nsd_id)[0]),
                        candidate=0, source_sha256=SOURCE_SHA, model_sha256=MODEL_SHA,
                        adapter_sha256=plan['adapter_sha256'],
                        png_sha256=sha(output), bytes=output.stat().st_size)
            savej(record, info); print(json.dumps(info), flush=True)
        savej(B / f'takagi-{pool}-complete.json', dict(plan, complete_utc=now(),
              gpu_inference_verified=True, generated_ids=ids.tolist(), output_directory=str(OUTPUT),
              identity_validation=str(validation_path), status='Generation only; eight metrics and atlas review pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', choices=('test100', 'testfull'), default='test100')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.generate:
        generate(args.pool, args.resume)
    else:
        plan = preflight(args.pool)[-1]
        savej(B / f'takagi-decoder-{args.pool}-preflight.json', plan)
        print(json.dumps(plan, indent=2))
