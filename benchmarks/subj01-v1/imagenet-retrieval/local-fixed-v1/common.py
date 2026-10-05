"""I/O and fixed definitions for the explicitly authorized local protocol."""
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BENCH = ROOT / 'benchmarks/subj01-v1'
OLD = ROOT.parent / 'neuroadapter-subject1-research'
ENC = OLD / 'repo/vendor/whole_brain_encoder'
DINO = OLD / 'repo/vendor/dinov2'
WEIGHTS = OLD / 'models/brain-encoder/dinov2_q_transformer/schaefer/subj_01'
ASSETS = ROOT / 'runtime/imagenet-retrieval-local-fixed-v1'
MASK = ROOT / 'runtime/imagenet-retrieval-assets/public-canonical-roi-union.npz'
RAW = OLD / 'data/raw/nsd'
DISCLOSURE = '本地固定协议；encoder开发集划分未核实；标准化及mask与作者历史实验的一致性未确认。'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(c)
    return h.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_json(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def limitations():
    return dict(encoder_split_verified=False, encoder_development_overlap='unknown',
                historical_beta_recipe_verified=False, historical_retrieval_mask_verified=False,
                result_scope='local_fixed_exploratory', disclosure=DISCLOSURE,
                training_updates=0, outbound_messages_sent=0)
