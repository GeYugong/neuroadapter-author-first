"""CPU-only native test inputs and frozen VD loading; no training or generation."""
import ast
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation must have no visible GPU.'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MPLCONFIGDIR'] = str(Path(__file__).resolve().parent / 'cache/mindbridge-matplotlib')

import numpy as np
import torch

B = Path(__file__).resolve().parent
R = B.parents[1]
SOURCE = B / 'upstream/mindbridge/src'
DEST = B / 'assets/mindbridge/data/webdataset_avg_split/test/subj01'
RECEIPT = B / 'mindbridge-inference-cpu-ready.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not RECEIPT.exists(), 'Completed preparation exists; read and reuse it.'
    assert DEST.resolve().is_relative_to(R.resolve())
    DEST.mkdir(parents=True, exist_ok=True)
    inputs = []
    for index in (0, 1):
        name = f'test_subj01_{index}.tar'
        path = B / 'assets/mindeye/webdataset_avg_split/test' / name
        expected = json.loads((B / 'assets' / f'verified-{name}.json').read_text())
        assert expected['verified'] and sha(path) == expected['sha256']
        count = 0
        with tarfile.open(path, 'r') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(('.nsdgeneral.npy', '.coco73k.npy')):
                    continue
                filename = Path(member.name).name
                assert filename.startswith('sample')
                raw = tar.extractfile(member).read()
                np.load(io.BytesIO(raw), allow_pickle=False)
                target = DEST / filename
                if target.exists():
                    assert target.read_bytes() == raw
                else:
                    with target.open('xb') as f:
                        f.write(raw)
                count += 1
        inputs.append(dict(tar=name, sha256=expected['sha256'], extracted_brain_and_id_files=count))
    # Original dataset loader: max-pool EACH repeat to8192 before recon.py takes mean.
    torch.set_num_threads(2)
    sys.path.insert(0, str(SOURCE))
    import data
    dataset = data.NSDDataset(str(DEST), extensions=['nsdgeneral.npy', 'coco73k.npy'],
                              pool_type='max', pool_num=8192)
    rows = []
    for index, key in enumerate(dataset.samples_keys):
        brain_path = DEST / (key + '.nsdgeneral.npy')
        raw = np.load(brain_path, allow_pickle=False)
        assert raw.shape == (3, 15724) and raw.dtype == np.float16
        pooled, coco = dataset[index]
        assert pooled.shape == (3, 8192) and torch.isfinite(pooled).all()
        image_id = int(coco.reshape(-1)[0])
        rows.append(dict(native_dataset_row=index, sample=key, nsd_id0=image_id,
                         brain_sha256=sha(brain_path), input_shape=[3, 15724], pooled_shape=[3, 8192]))
    ids = {row['nsd_id0'] for row in rows}
    locked = set(np.loadtxt(B / 'test_full_ids.txt', dtype=int).tolist())
    assert len(rows) == len(ids) == 982 and ids == locked
    training = json.loads((B / 'mindbridge-data-validation.json').read_text())
    exposed = {row['nsd_id0'] for split in ('train', 'val') for row in training['splits'][split]}
    assert ids.isdisjoint(exposed)
    mapping = B / 'mindbridge-test-inputs.json'
    mapping.write_text(json.dumps(dict(source_revision=training['source_revision'],
        archives=inputs, n_test=982, ids_match_locked_test=True, train_val_disjoint=True,
        preprocessing='Original NSDDataset float32/max-pool each of3 repeats to8192; subsequent reconstruction averages repeats',
        image_files_extracted=False, rows=rows), indent=2) + '\n')

    # Execute the original prepare_VD, only redirect its scheduler asset lookup.
    from diffusers import UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    original = UniPCMultistepScheduler.from_pretrained
    def local_scheduler(cls, name, *args, **kwargs):
        assert name == 'shi-labs/versatile-diffusion'
        kwargs.pop('cache_dir', None)
        kwargs.pop('subfolder', None)
        return original(str(B / 'assets/versatile-diffusion/scheduler'), *args, **kwargs)
    UniPCMultistepScheduler.from_pretrained = classmethod(local_scheduler)
    source = SOURCE / 'recon.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare_VD')
    ns = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
    pipeline = ns['prepare_VD'](SimpleNamespace(vd_cache_dir=str(B / 'assets/versatile-diffusion'),
                                               text_image_ratio=.5), torch.device('cpu'))
    duals = [m for m in pipeline.image_unet.modules() if isinstance(m, DualTransformer2DModel)]
    assert duals
    for module in duals:
        assert module.mix_ratio == .5 and list(module.condition_lengths) == [77, 257]
        assert list(module.transformer_index_for_condition) == [1, 0]
    assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in pipeline.image_unet.parameters())
    assert not any(p.requires_grad for p in pipeline.image_unet.parameters())
    assert not any(p.requires_grad for p in pipeline.vae.parameters())
    assert not torch.cuda.is_initialized()
    out = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        native_test_inputs=982, input_mapping_sha256=sha(mapping), train_val_disjoint=True,
        original_prepare_VD_cpu_loaded=True, recon_source_sha256=sha(source),
        scheduler=type(pipeline.scheduler).__name__, dual_transformers=len(duals),
        image_text_mix=.5, precision='FP32, original from_pretrained default',
        image_unet_parameters=sum(p.numel() for p in pipeline.image_unet.parameters()),
        cuda_initialized=False, training_updates=0, generated_candidates=0,
        checkpoint_policy='Original inference.sh last.pth, after full600 training and complete exit; never select by test',
        native_sampling='inference.sh CFG5, options.py20steps, seed42; K1 benchmark replaces native8',
        lowlevel_policy='Original inference.sh does not supply autoencoder_name; no optional MindEye autoencoder grafted into Vanilla',
        ground_truth_policy='Original utils.reconstruction selects native candidates using CLIP of GT image; this selector/plotting must be omitted for K1 and is disallowed in a GT-free native-selection table',
        status='CPU assets and input mapping ready;600epoch checkpoint and actual GPU K1 identity/generation/evaluation still pending')
    RECEIPT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
