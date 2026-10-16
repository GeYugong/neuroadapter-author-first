"""Pinned MindBridge Vanilla reconstruction and published GT-CLIP eight-way selector.

Run with runtime/env/bin/python -B. Never changes existing K1 outputs or trains.
"""
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time

R = Path('/data1/matengyu/geyugong/neuroadapter-author-first')
B = R / 'benchmarks/subj01-v1'
O = B / 'outputs/mindbridge-native8'
A = R / 'reports/native-protocols-20261016/mindbridge'
S = B / 'upstream/mindbridge/src'
W = B / 'upstream/mindbridge/train_logs/benchmark-subj01-vanilla-seed42/last.pth'
HASHES = {'models.py': '5f2481b82fdcb547a9b9ce1902290930312de3301a0e17a7dca721ca82873e13',
          'utils.py': 'cdf5f71c5a0c8d30b3dc3deb16e43f0b019f7abbda524bff4825c2c840021a99',
          'data.py': 'cbae812311894ce24cdb22bfcccda05674c0016fdadb7138c5777f441a25719f',
          'recon.py': '25d51784be7fcdda765df60ec30c11ba7ea272162be129cf98a21705b07deb5c'}
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4')
sys.path.insert(0, str(B))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj, indent=2)+'\n')
    t.replace(p)


def sources():
    for f, digest in HASHES.items(): assert sha(S/f) == digest, f
    assert sha(W) == '5bd75a992755d70183633df514754b9084b0a4a681284afc83621b2387aac9ad'
    done = json.loads((B/'mindbridge-training-complete.json').read_text())
    assert done['completed_epochs'] == 600 and done['full_state_verified']
    ids = [int(x) for x in (B/'test_full_ids.txt').read_text().split()]
    rows = json.loads((B/'mindbridge-test-inputs.json').read_text())['rows']
    rows = {r['nsd_id0']: r for r in rows}
    assert len(ids) == len(set(ids)) == len(rows) == 982 and set(ids) == set(rows)
    return ids, rows


def compile_official():
    import torch
    import numpy as np
    import clip
    from torchvision import transforms
    ns = dict(torch=torch, nn=torch.nn, np=np, clip=clip, transforms=transforms,
              math=math, random=random, os=os, F=torch.nn.functional)
    definitions = {'models.py': ['Adapter_Layer', 'ResMLP', 'MindSingle', 'Clipper'],
                   'utils.py': ['reconstruction', 'batchwise_cosine_similarity', 'decode_latents'],
                   'data.py': ['pool_voxels']}
    for file, names in definitions.items():
        tree = ast.parse((S/file).read_text())
        nodes = [n for n in tree.body if getattr(n, 'name', None) in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(S/file), 'exec'), ns)
    return ns


def prepare():
    ids, rows = sources()
    A.mkdir(parents=True, exist_ok=True)
    jpg = A/'selector-jpg'; jpg.mkdir(exist_ok=True)
    expected = {r['sample']+'.jpg' for r in rows.values()}
    found = set()
    for index in range(2):
        path = B/f'assets/mindeye/webdataset_avg_split/test/test_subj01_{index}.tar'
        with tarfile.open(path) as archive:
            for entry in archive:
                if entry.name not in expected: continue
                assert entry.isfile() and Path(entry.name).name == entry.name
                assert entry.name not in found
                raw = archive.extractfile(entry).read()
                target = jpg/entry.name
                if target.exists(): assert target.read_bytes() == raw
                else: target.write_bytes(raw)
                found.add(entry.name)
    assert found == expected
    inp = B/'assets/mindbridge/data/webdataset_avg_split/test/subj01'
    import numpy as np
    for i in ids:
        row = rows[i]
        assert sha(inp/(row['sample']+'.nsdgeneral.npy')) == row['brain_sha256']
        assert int(np.load(inp/(row['sample']+'.coco73k.npy')).reshape(-1)[0]) == i
    compile_official()
    save(A/'preflight.json', dict(status='cpu_inputs_and_source_passed', n=982,
         source_sha256=HASHES, checkpoint_sha256=sha(W), K=8,
         selection='Original GT image hidden-token CLIP cosine argmax; GT is used for selection',
         jpg_sha256={n:sha(jpg/n) for n in sorted(found)}, steps=20, cfg=5,
         seed=42, lowlevel=False, mix_ratio=.5, precision='FP32',
         gpu_forward_verified=False, runtime_compatibility='Unmodified official AST definitions; deprecated VD module alias only'))


def free_gpu(preferred=None):
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'], text=True)
    jobs = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
    for line in raw.splitlines():
        index, uuid, memory = [x.strip() for x in line.split(',')]
        if preferred is not None and index != str(preferred): continue
        if int(memory)<512 and uuid not in jobs: return index
    return None


def generate(recovery2=False):
    ids, rows = sources()
    assert (A/'preflight.json').exists()
    assert free_gpu(os.environ.get('CUDA_VISIBLE_DEVICES')) == os.environ.get('CUDA_VISIBLE_DEVICES'), 'Selected GPU no longer free'
    marker = 'generation-recovery2-started.json' if recovery2 else 'generation-started.json'
    assert not (A/marker).exists(), 'Inspect failure before an explicit recovery adaptation'
    assert shutil.disk_usage(R).free > 40*2**30
    O.mkdir(parents=True, exist_ok=True)
    assert set(p.name for p in O.iterdir()) <= ({'protocol.json'} if recovery2 else set())
    import torch
    import numpy as np
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    device = torch.device('cuda:0')
    save(A/marker, dict(pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'], runner_sha256=sha(__file__)))
    import diffusers
    module = importlib.import_module('diffusers.pipelines.deprecated.versatile_diffusion')
    module.ModelMixin = diffusers.ModelMixin
    sys.modules['versatile_diffusion'] = module
    from diffusers.models import DualTransformer2DModel
    original_forward = DualTransformer2DModel.forward
    def compatible_forward(self, *args, encoder_attention_mask=None, **kwargs):
        assert encoder_attention_mask is None, 'Never discard a real attention mask'
        return original_forward(self, *args, **kwargs)
    DualTransformer2DModel.forward = compatible_forward
    from mindbridge_local_assets import configure
    configure()
    from mindbridge_reconstruct_k1 import load_vd
    ns = compile_official()
    pipe = load_vd(torch, device)
    model = ns['MindSingle'](in_dim=8192, out_dim_image=257*768, out_dim_text=77*768,
                             h=2048, n_blocks=4, subj_list=[1])
    checkpoint = torch.load(W, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 599
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    model.eval().requires_grad_(False).to(device)
    extractor = ns['Clipper']('ViT-L/14', hidden_state=True, norm_embs=True, device=device)
    scores = []
    original_cosine = ns['batchwise_cosine_similarity']
    def capture(z, b):
        value = original_cosine(z, b); scores.append(value.item()); return value
    ns['batchwise_cosine_similarity'] = capture
    save(O/'protocol.json', json.loads((A/'preflight.json').read_text()))
    for index, i in enumerate(ids):
        assert shutil.disk_usage(R).free > 15*2**30
        row = rows[i]
        brain_path = B/'assets/mindbridge/data/webdataset_avg_split/test/subj01'/(row['sample']+'.nsdgeneral.npy')
        voxel = ns['pool_voxels'](torch.from_numpy(np.load(brain_path)),8192,'max').mean(0,keepdim=True).to(device)
        gt = np.array(Image.open(A/'selector-jpg'/(row['sample']+'.jpg')).convert('RGB')).astype(np.float32)/255
        gt = torch.from_numpy(gt.transpose(2,0,1)).unsqueeze(0)
        scores.clear()
        _, candidates, picks, _ = ns['reconstruction'](gt, voxel, model, extractor,
             pipe.image_unet, pipe.vae, pipe.scheduler, img_lowlevel=None,
             num_inference_steps=20, recons_per_sample=8, guidance_scale=5,
             img2img_strength=1, seed=42, plotting=False, n_samples_save=1,
             device=device, mem_efficient=False)
        assert candidates.shape == (1,8,3,512,512) and candidates.dtype == torch.float32
        assert torch.isfinite(candidates).all() and len(scores)==8 and np.isfinite(scores).all()
        pick = int(picks[0]); assert pick == int(np.argmax(scores))
        tensor = O/f'{i}.candidates.pt'; torch.save(candidates, tensor)
        png = O/f'{i}.png'; to_pil_image(candidates[0,pick]).save(png)
        save(O/f'{i}.json',dict(nsd_id0=i,K=8,candidate=pick,seed=42,
             selection='GT hidden-token CLIP cosine argmax',scores=list(scores),
             candidates_path=tensor.name,candidates_sha256=sha(tensor),prediction_sha256=sha(png)))
        save(A/'generation-progress.json',dict(completed=index+1,total=982,last_id=i,
             peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        print(json.dumps(dict(completed=index+1,nsd_id0=i,pick=pick)),flush=True)
    save(A/'generation-complete.json',dict(n=982,status='complete'))


def controller(recovery=False):
    assert (A/'preflight.json').exists()
    prefix = ('recovery2-' if recovery == 2 else 'recovery1-') if recovery else ''
    if recovery == 2:
        assert json.loads((A/'recovery1-generation-exit.json').read_text())['exit_code'] == 1
        assert "unexpected keyword argument 'encoder_attention_mask'" in (A/'recovery1-generate.log').read_text()
        assert set(p.name for p in O.iterdir()) == {'protocol.json'}
    elif recovery:
        assert not (A/'generation-started.json').exists() and not O.exists()
        assert json.loads((A/'generation-exit.json').read_text())['exit_code'] == 1
        assert 'Selected GPU no longer free' in (A/'generate.log').read_text()
    assert not (A/(prefix+'controller-started.json')).exists()
    save(A/(prefix+'controller-started.json'),dict(pid=os.getpid(),runner_sha256=sha(__file__),recovery_before_any_generation=recovery))
    while True:
        status = R/'reports/l450-best8-test982/score-recovery2-status.json'
        ready = status.exists() and json.loads(status.read_text()).get('status') == 'metrics_complete'
        gpu = free_gpu() if ready else None
        if gpu is not None: break
        save(A/'controller-status.json',dict(status='waiting_l450_metrics' if not ready else 'waiting_free_gpu',pid=os.getpid()))
        time.sleep(30)
    env = os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
    save(A/'controller-status.json',dict(status='generating',gpu=gpu,pid=os.getpid(),attempt=prefix))
    with (A/(prefix+'generate.log')).open('x') as log:
        code=subprocess.call([str(R/'runtime/env/bin/python'),'-B',str(Path(__file__).resolve()),'generate-recovery2' if recovery == 2 else 'generate'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(A/(prefix+'generation-exit.json'),dict(exit_code=code))
    if code != 0:
        save(A/'controller-status.json',dict(status='failed',exit_code=code)); return
    source = (B/'evaluate_native16.py').read_text()
    for old,new in [('mindeye1-native16','mindbridge-native8'), ('==16','==8'), ('K=16','K=8'),
                    ('== 16','== 8'), ('<16','<8'), ('(1,16,3,512,512)','(1,8,3,512,512)'),
                    ("'K':16","'K':8"), ('author brain-predicted CLIP similarity; no GT','author GT hidden-token CLIP similarity'),
                    ('native brain-only selector','native GT-CLIP selector')]:
        source = source.replace(old,new)
    # Keep __file__ at the benchmark location used by the unmodified evaluator.
    wrapper = 'import sys\nsys.path.insert(0,'+repr(str(B))+')\n__file__='+repr(str(B/'evaluate_native16.py'))+'\n'+source
    evaluate = A/'evaluate-native8.py'; evaluate.write_text(wrapper)
    with (A/'score.log').open('x') as log:
        code = subprocess.call([str(R/'runtime/env/bin/python'),'-B',str(evaluate),'--method','mindbridge-native8',
                    '--ids',str(B/'test_full_ids.txt'),'--feature-batch','16'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(A/'score-exit.json',dict(exit_code=code))
    save(A/'controller-status.json',dict(status='metrics_complete_needs_review' if code==0 else 'score_failed',exit_code=code))


if __name__ == '__main__':
    {'prepare':prepare,'generate':generate,'controller':controller,'recover-before-generation':lambda:controller(True),
     'generate-recovery2':lambda:generate(True),'recover-empty-mask':lambda:controller(2)}[sys.argv[1]]()
