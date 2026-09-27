"""Validate complete author MindSingle/Clipper on CPU, without data or updates."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B / 'upstream/mindbridge/src'))
import torch
import clip
torch.set_num_threads(4)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


url = clip.clip._MODELS['ViT-L/14']
expected = url.split('/')[-2]
cached = Path.home() / '.cache/clip/ViT-L-14.pt'
assert cached.is_file() and sha(cached) == expected, 'Existing official CLIP cache identity mismatch'
owned = B / 'assets/mindbridge/clip/ViT-L-14.pt'
owned.parent.mkdir(parents=True, exist_ok=True)
if not owned.exists():
    temp = owned.with_suffix('.pt.tmp')
    shutil.copyfile(cached, temp)
    assert sha(temp) == expected
    temp.replace(owned)
else:
    assert sha(owned) == expected

from mindbridge_local_assets import configure
configure()
from models import Clipper, MindSingle
extractor = Clipper('ViT-L/14', hidden_state=True, norm_embs=True, device=torch.device('cpu'))
assert all(not p.requires_grad for p in extractor.parameters())
model = MindSingle(in_dim=8192, out_dim_image=257*768, out_dim_text=77*768,
                   h=2048, n_blocks=4, subj_list=[1])
parameters = sum(p.numel() for p in model.parameters())
assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in model.parameters())
report = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'CPU-only complete author model and frozen CLIP load; no images, scores, gradients, or training updates',
    'source_commit': '2a3943ff5b619db6f85f16a5e7ff1331d25def55',
    'variant': 'MindSingle Subject1 Vanilla', 'h_size': 2048, 'n_blocks': 4, 'pool_num': 8192,
    'trainable_parameter_count': parameters, 'dtype': 'float32',
    'full_author_Clipper_loaded': True, 'CLIP_frozen': True,
    'openai_CLIP_url': url, 'openai_CLIP_sha256': expected,
    'HF_CLIP_sha256': sha(B / 'assets/clip-vit-large-patch14/pytorch_model.bin'),
    'engineering_adaptation': 'Exact official model-name resolution redirected to existing local assets; no model/preprocessing methods replaced.',
    'CPU_initialization_is_discarded': True, 'training_initialization': 'Original main.py seed42 during actual training; CPU probe weights will never be resumed.',
    'new_training_updates': 0, 'GPU_memory_or_DeepSpeed_not_yet_validated': True,
}
(B / 'mindbridge-model-cpu-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
