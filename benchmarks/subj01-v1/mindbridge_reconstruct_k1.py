"""Guarded native Subject1 Vanilla K1 runner; default is CPU-only preflight.

Requires an independently audited full600 completion receipt before generation.
No ground truth is loaded or accepted by the generation function.
"""
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from mindbridge_k1_native import compile_native

B = Path(__file__).resolve().parent
SOURCE = B / 'upstream/mindbridge/src'
TRAIN = SOURCE.parent / 'train_logs/benchmark-subj01-vanilla-seed42'
OUTPUT = B / 'outputs/mindbridge'
STATE = B / 'outputs/mindbridge-records'
INPUT = B / 'assets/mindbridge/data/webdataset_avg_split/test/subj01'
RECON_SHA = '25d51784be7fcdda765df60ec30c11ba7ea272162be129cf98a21705b07deb5c'
ID_SHAS = {'test100': '63380f171be2852c280fc7c0c2737195e25c808adf4bf9c10c23981b4961732a',
           'testfull': '4c438772ffa978ab8c5f781b8a3fc73df0159ae4892dc39d896f05a413d36441'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def savej(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def preflight(pool):
    import torch
    import numpy as np
    assert not torch.cuda.is_initialized()
    ready = json.loads((B / 'mindbridge-inference-cpu-ready.json').read_text())
    mapping = B / 'mindbridge-test-inputs.json'
    assert sha(mapping) == ready['input_mapping_sha256']
    data = json.loads(mapping.read_text())
    assert data['train_val_disjoint'] and data['ids_match_locked_test']
    rows = {r['nsd_id0']: r for r in data['rows']}
    assert len(rows) == 982
    idsfile = B / ('test100_ids.txt' if pool == 'test100' else 'test_full_ids.txt')
    assert sha(idsfile) == ID_SHAS[pool]
    ids = [int(x) for x in idsfile.read_text().split()]
    assert len(ids) == len(set(ids)) == (100 if pool == 'test100' else 982)
    full = B / 'test_full_ids.txt'
    assert sha(full) == ID_SHAS['testfull']
    assert set(rows) == {int(x) for x in full.read_text().split()}
    assert sha(SOURCE / 'recon.py') == RECON_SHA
    assert subprocess.check_output(['git', '-C', str(SOURCE.parent), 'rev-parse', 'HEAD'], text=True).strip() == '2a3943ff5b619db6f85f16a5e7ff1331d25def55'
    assert not subprocess.check_output(['git', '-C', str(SOURCE.parent), 'diff', 'HEAD', '--', 'src'], text=True)
    _, _, audit = compile_native(torch)
    # Validate every requested raw repeat and ID; never open test JPG or GT.
    for image_id in ids:
        row = rows[image_id]
        path = INPUT / (row['sample'] + '.nsdgeneral.npy')
        assert sha(path) == row['brain_sha256']
        raw = np.load(path, allow_pickle=False)
        assert raw.shape == (3, 15724) and raw.dtype == np.float16 and np.isfinite(raw).all()
        actual = np.load(INPUT / (row['sample'] + '.coco73k.npy'), allow_pickle=False)
        assert int(actual.reshape(-1)[0]) == image_id
    result = dict(utc=now(), pool=pool, n_eval=len(ids), ids_sha256=ID_SHAS[pool],
                  mapping_sha256=sha(mapping), runner_sha256=sha(Path(__file__).resolve()),
                  generation_module_sha256=sha(B / 'mindbridge_k1_native.py'), generation_audit=audit,
                  recon_source_sha256=RECON_SHA, K=1, selection='none',
                  preprocessing='Original NSDDataset float32/maxpool each repeat8192, then mean3',
                  scheduler='Original UniPC', cfg=5, steps=20, seed=42, text_image_ratio=.5,
                  precision='FP32, original VD default and no autocast', lowlevel=False,
                  gt_access=False, real_model_inference=False)
    return ids, rows, result


def readiness():
    receipt = B / 'mindbridge-training-complete.json'
    assert receipt.exists(), 'Full600 training audit receipt missing; training is not inference-ready.'
    record = json.loads(receipt.read_text())
    assert record['completed_epochs'] == 600 and record['global_steps'] == 51600
    assert record['resume_exit_code'] == 0 and record['full_state_verified'] is True
    assert (B / 'logs/mindbridge-resume.exit').read_text().strip() == '0'
    meta = json.loads((TRAIN / 'last/trainer-recovery.json').read_text())
    assert meta['completed_epoch_zero_based'] == 599 and meta['next_epoch'] == 600
    expected = record['checkpoint_sha256']
    assert sha(TRAIN / 'last.pth') == expected
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in ('0', '1'), 'Select exactly one explicit physical GPU.'
    uuid = subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(line.startswith(uuid) for line in jobs.splitlines()), 'Selected GPU occupied; do not evict jobs.'
    return expected, physical


def load_vd(torch, device):
    from diffusers import UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    original = UniPCMultistepScheduler.from_pretrained

    def local(cls, name, *args, **kwargs):
        assert name == 'shi-labs/versatile-diffusion'
        kwargs.pop('cache_dir', None)
        kwargs.pop('subfolder', None)
        return original(str(B / 'assets/versatile-diffusion/scheduler'), *args, **kwargs)

    UniPCMultistepScheduler.from_pretrained = classmethod(local)
    tree = ast.parse((SOURCE / 'recon.py').read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_VD')
    ns = {}
    try:
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(SOURCE / 'recon.py'), 'exec'), ns)
        pipe = ns['prepare_VD'](SimpleNamespace(vd_cache_dir=str(B / 'assets/versatile-diffusion'), text_image_ratio=.5), device)
    finally:
        UniPCMultistepScheduler.from_pretrained = original
    duals = [m for m in pipe.image_unet.modules() if isinstance(m, DualTransformer2DModel)]
    assert len(duals) == 16
    assert all(m.mix_ratio == .5 and list(m.condition_lengths) == [77, 257]
               and list(m.transformer_index_for_condition) == [1, 0] for m in duals)
    assert all(p.dtype == torch.float32 for p in pipe.image_unet.parameters())
    return pipe


class UnusedClip:
    def to(self, device):
        return self

    def embed_image(self, *args, **kwargs):
        raise AssertionError('CLIP GT selection is forbidden')


def generate(pool, resume):
    # Guard before model loading or output creation, including when training is live.
    checkpoint_sha, physical = readiness()
    ids, rows, plan = preflight(pool)
    if pool == 'testfull':
        publication = json.loads((B / 'mindbridge-test100-publication.json').read_text())
        assert publication['n_eval'] == 100 and publication['atlas_images_reviewed'] == 32
        assert publication['checkpoint_sha256'] == checkpoint_sha
        assert publication['published_commit']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = STATE / (pool + '.started.json')
        assert not (B / f'mindbridge-{pool}-complete.json').exists(), 'Pool already complete'
        assert resume or not marker.exists(), 'Inspect earlier attempt; explicit --resume required'
        attempt = dict(plan, checkpoint_sha256=checkpoint_sha, physical_gpu=physical, pid=os.getpid())
        if marker.exists():
            previous = json.loads(marker.read_text())
            for key in ('runner_sha256', 'generation_module_sha256', 'checkpoint_sha256', 'ids_sha256'):
                assert previous[key] == attempt[key], key
            attempt['previous_attempts'] = previous.get('previous_attempts', []) + [{k: v for k, v in previous.items() if k != 'previous_attempts'}]
        savej(marker, attempt)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          MPLCONFIGDIR=str(B / 'cache/mindbridge-matplotlib'))
        import torch
        import numpy as np
        from PIL import Image
        torch.set_num_threads(4)
        sys.path.insert(0, str(SOURCE))
        import utils
        import data
        from models import MindSingle
        utils.seed_everything(42)
        device = torch.device('cuda:0')
        pipe = load_vd(torch, device)
        model = MindSingle(in_dim=8192, out_dim_image=257 * 768, out_dim_text=77 * 768,
                           h=2048, n_blocks=4, subj_list=[1])
        checkpoint = torch.load(TRAIN / 'last.pth', map_location='cpu')
        assert checkpoint['epoch'] == 599
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        del checkpoint
        model.requires_grad_(False).eval().to(device)
        dataset = data.NSDDataset(str(INPUT), extensions=['nsdgeneral.npy', 'coco73k.npy'],
                                  pool_type='max', pool_num=8192)
        k1, reference, _ = compile_native(torch)
        identity_path = STATE / 'gpu-prefix-identity.json'
        for image_id in ids:
            row = rows[image_id]
            imagefile = OUTPUT / f'{image_id}.png'
            tensorfile = OUTPUT / f'{image_id}.pt'
            recordfile = STATE / f'{image_id}.json'
            if recordfile.exists():
                record = json.loads(recordfile.read_text())
                for key in ('runner_sha256', 'generation_module_sha256', 'checkpoint_sha256'):
                    assert record[key] == attempt[key], key
                assert record['brain_sha256'] == row['brain_sha256'] and record['nsd_id0'] == image_id
                assert imagefile.exists() and sha(imagefile) == record['output_sha256']
                assert tensorfile.exists() and sha(tensorfile) == record['native_tensor_sha256']
                continue
            assert not imagefile.exists() and not tensorfile.exists(), 'Orphan output: inspect before resuming, never overwrite'
            index = row['native_dataset_row']
            assert dataset.samples_keys[index] == row['sample']
            repeats, actual_id = dataset[index]
            assert int(actual_id.reshape(-1)[0]) == image_id and repeats.shape == (3, 8192)
            voxel = repeats.unsqueeze(0).mean(axis=1).float().to(device)
            if not identity_path.exists():
                assert image_id == int((B / 'test100_ids.txt').read_text().split()[0])
                # Original generation prefix only, with a synthetic placeholder,
                # not GT. Capture actual-model equivalence before accepting output.
                with torch.no_grad():
                    original = reference(torch.zeros(1, 3, 1, 1, device=device), voxel, model,
                        UnusedClip(), pipe.image_unet, pipe.vae, pipe.scheduler,
                        img_lowlevel=None, num_inference_steps=20, recons_per_sample=1,
                        guidance_scale=5., img2img_strength=1., seed=42, plotting=False,
                        verbose=False, n_samples_save=1, device=device, mem_efficient=False)
                    generated = k1(voxel, model, pipe.image_unet, pipe.vae, pipe.scheduler, device=device)
                assert torch.equal(original, generated), 'Actual GPU native-prefix identity failed'
                savej(identity_path, dict(utc=now(), nsd_id0=image_id, checkpoint_sha256=checkpoint_sha,
                    generation_module_sha256=attempt['generation_module_sha256'], tensor_identical=True,
                    placeholder='Synthetic zero tensor only, no GT loaded', extra_identity_candidates=1))
                del original
            else:
                identity = json.loads(identity_path.read_text())
                assert identity['checkpoint_sha256'] == checkpoint_sha and identity['tensor_identical']
                assert identity['generation_module_sha256'] == attempt['generation_module_sha256']
                with torch.no_grad():
                    generated = k1(voxel, model, pipe.image_unet, pipe.vae, pipe.scheduler, device=device)
            assert generated.shape == (1, 1, 3, 512, 512) and torch.isfinite(generated).all()
            # Original recon.py saves this float tensor. Keep it for evaluation;
            # the uint8 PNG is only an atlas/display derivative.
            temp_tensor = tensorfile.with_suffix('.tmp.pt')
            torch.save(generated, temp_tensor)
            os.replace(temp_tensor, tensorfile)
            pixels = (generated[0, 0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            temp = imagefile.with_suffix('.tmp.png')
            Image.fromarray(pixels).save(temp)
            os.replace(temp, imagefile)
            savej(recordfile, dict(utc=now(), nsd_id0=image_id, native_dataset_row=index,
                brain_sha256=row['brain_sha256'], checkpoint_sha256=checkpoint_sha,
                runner_sha256=attempt['runner_sha256'], generation_module_sha256=attempt['generation_module_sha256'],
                output_sha256=sha(imagefile), native_tensor_sha256=sha(tensorfile),
                evaluation_input='Original FP32 reconstruction tensor, no PNG quantization',
                K=1, selection='none', gt_access=False))
            print(json.dumps(dict(utc=now(), generated_id=image_id)), flush=True)
        savej(B / f'mindbridge-{pool}-complete.json', dict(attempt, utc_complete=now(),
            real_model_inference=True, generated_or_verified=len(ids),
            extra_identity_candidates=1, status='Generation only; scoring, atlas and publication still required'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', choices=['test100', 'testfull'], default='test100')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.generate:
        generate(args.pool, args.resume)
    else:
        _, _, report = preflight(args.pool)
        assert not args.resume
        savej(B / f'mindbridge-decoder-{args.pool}-preflight.json', report)
        print(json.dumps(report, indent=2))
