"""Read-only final result verification and fixed-ID contact sheets for Subject 2."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

R = Path(__file__).resolve().parents[1]
P = R / 'reports/subject02-bf16-seed42'
O = P / 'test982-best8'
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / 'scripts'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  MPLCONFIGDIR=str(R / 'runtime/matplotlib'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    assert path.resolve().is_relative_to(O.resolve()) and not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def audit():
    import numpy as np
    import pandas as pd
    import torch
    from brain_adapter.dataset import nsd_topk_parcel_dataset
    from subject02_test_best8 import frozen, seed_for

    torch.set_num_threads(4)
    plan = frozen()
    assert plan['selected_epoch'] == 109
    ids = np.loadtxt(R / 'data/subject02/standard-test-ids.txt', dtype=int)
    assert len(ids) == len(set(ids)) == 982
    names = {f'image_{i:06d}.npz' for i in ids}
    for folder in ['candidates', 'k1', 'selected']:
        assert {f.name for f in (O / folder).iterdir()} == names
    assert {f.name for f in (O / 'records').iterdir()} == {f'{i}.json' for i in ids}
    assert not list((O / 'queue').iterdir())
    assert not list(O.glob('*failure*.json'))
    exits = {s: json.loads((O / f'{s}-exit.json').read_text())['returncode']
             for s in ['generate', 'select', 'score']}
    assert set(exits.values()) == {0}
    cfg = json.loads((P / 'effective-config.json').read_text())
    ds = nsd_topk_parcel_dataset(SimpleNamespace(subj=2, hemi=None, backbone_arch='dinov2_q',
        data_dir=cfg['data_dir'], imgs_dir=cfg['imgs_dir'], parcel_dir=cfg['parcel_dir'],
        tokenizer=None, gen_size=512), split='test', topk=100)
    vertices = torch.load(R / plan['weight'].rsplit('/', 1)[0] / 'parcel_vertices.pt', weights_only=True)
    actual = ds.get_selected_voxel_indices()
    for hemi in ['lh', 'rh']:
        assert len(vertices[hemi]) == len(actual[hemi]) == 100
        assert all(torch.equal(x, y) for x, y in zip(vertices[hemi], actual[hemi]))
    lookup = {int(i): j for j, i in enumerate(ds.base_dataset.split_presented_imgs)}
    train = set(np.loadtxt(R / 'data/subject02/train_ids.txt', dtype=int))
    val = set(np.loadtxt(R / 'data/subject02/val_ids.txt', dtype=int))
    assert not (set(ids) & (train | val)) and not (train & val)
    seeds_seen = set()
    pixels_seen = set()
    hist = collections.Counter()
    records_digest = hashlib.sha256()
    for position, image_id in enumerate(ids):
        image_id = int(image_id)
        record_path = O / 'records' / f'{image_id}.json'
        rec = json.loads(record_path.read_text())
        records_digest.update(f'{image_id}:{sha(record_path)}\n'.encode())
        seeds = [seed_for(image_id, c) for c in range(8)]
        assert rec['image_id'] == image_id and rec['seeds'] == seeds
        assert len(set(seeds)) == 8 and not seeds_seen.intersection(seeds)
        seeds_seen.update(seeds)
        scores = np.asarray(rec['correlations'])
        assert scores.shape == (8,) and np.isfinite(scores).all()
        winner = int(scores.argmax())
        assert rec['selected_index'] == winner and rec['GT_image_used_for_selection'] is False
        hist[winner] += 1
        filename = f'image_{image_id:06d}.npz'
        for folder, key in [('candidates', 'candidate_archive_sha256'),
                            ('selected', 'selected_npz_sha256'), ('k1', 'k1_npz_sha256')]:
            assert sha(O / folder / filename) == rec[key]
        with np.load(O / 'candidates' / filename) as a, np.load(O / 'selected' / filename) as s, \
                np.load(O / 'k1' / filename) as k:
            assert int(a['image_id']) == int(s['image_id']) == int(k['image_id']) == image_id
            images = a['candidate_images']
            assert images.shape == (8, 512, 512, 3) and images.dtype == np.uint8
            pixels = [hashlib.sha256(x.tobytes()).hexdigest() for x in images]
            assert pixels == rec['candidate_pixel_sha256'] and len(set(pixels)) == 8
            pixels_seen.update(pixels)
            assert a['seeds'].tolist() == seeds and s['all_seeds'].tolist() == seeds
            assert s['seeds'].tolist() == [seeds[winner]] and k['seeds'].tolist() == seeds[:1]
            assert np.array_equal(s['brain_correlations'], scores) and int(s['selected_index']) == winner
            assert np.array_equal(s['candidate_images'][0], images[winner])
            assert np.array_equal(k['candidate_images'][0], images[0])
            row = ds[lookup[image_id]]
            parts = []
            for hemi in ['lh', 'rh']:
                key = f'brain_{hemi}_f'
                assert np.array_equal(a[key][0], row[key].numpy())
                assert np.isfinite(a[key]).all()
                parts.extend([a[key][:, j, :len(v)] for j, v in enumerate(vertices[hemi])])
            brain = np.concatenate(parts, axis=1)
            assert hashlib.sha256(brain.tobytes()).hexdigest() == rec['measured_brain_sha256']
            assert len(ds.base_dataset.img_to_runs[lookup[image_id]]) == 3
        if (position + 1) % 100 == 0:
            print('AUDITED', position + 1, flush=True)
    assert len(seeds_seen) == 7856 and len(pixels_seen) == 7856
    metrics = json.loads((O / 'metrics.json').read_text())
    assert metrics['n'] == metrics['negative_pool'] == 982 and metrics['feature_batch'] == 16
    means_error = {}
    for label in ['K1', 'best8']:
        frame = pd.read_csv(O / f'{label}-per-image.csv')
        assert frame['image_id'].tolist() == ids.tolist() and len(frame) == 982
        assert np.isfinite(frame.to_numpy()).all()
        errors = {m: abs(float(frame[m].mean()) - v) for m, v in metrics['metrics'][label].items()}
        assert max(errors.values()) < 1e-10
        means_error[label] = errors
    for m, delta in metrics['delta'].items():
        assert abs(delta - (metrics['metrics']['best8'][m] - metrics['metrics']['K1'][m])) < 1e-12
    save(O / 'final-audit.json', dict(complete=True, time=time.time(), n=982, K=8,
        candidate_images_verified=7856, unique_seeds=7856, unique_candidate_pixel_hashes=7856,
        exact_ID_sets=True, all_argmax_verified=True, K1_exact_candidate0=True,
        all_archive_hashes_verified=True, all_measured_brain_matches_source_dataset=True,
        full40_sessions_three_repetitions_per_ID=True, train_val_test_disjoint=True,
        full982_negative_pool=True, feature_batch_only=16, per_image_mean_abs_errors=means_error,
        protected_sha256_count=len(plan['protected_sha256']), protected_files_unchanged=True,
        large_data_file_stat_guards_verified=len(plan['data_files']), stage_exit_codes=exits,
        records_ordered_digest=records_digest.hexdigest(), selected_index_counts=dict(hist),
        selected_epoch=109, K4_performed=False, test_used_to_switch_checkpoint=False,
        metrics_sha256=sha(O / 'metrics.json'), audit_code_sha256=sha(Path(__file__)),
        pixel_data_unmodified=True, visual_review_pending=True))
    print('AUDIT_COMPLETE', flush=True)


def atlas():
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    plan = json.loads((O / 'plan.json').read_text())
    assert plan['fixed_visual_ids'] == 'first32 IDs in verified Subject2 standard-test-ids.txt order'
    ids = np.loadtxt(R / 'data/subject02/standard-test-ids.txt', dtype=int)[:32].tolist()
    target = O / 'review-atlas'
    assert not target.exists()
    target.mkdir()
    cfg = json.loads((P / 'effective-config.json').read_text())
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 18)
    title_font = ImageFont.truetype(font_path, 24)
    manifest = dict(ids=ids, rule='First32 standard test IDs; fixed before inference; no reselection',
                    source_metrics_sha256=sha(O / 'metrics.json'), pages=[],
                    presentation_only='Uniform full-frame resizing; no crop, enhancement or content editing')
    with h5py.File(Path(cfg['imgs_dir']) / 'nsd_stimuli.hdf5', 'r') as h5:
        for page in range(4):
            # Three-column comparisons retain larger thumbnails for detailed visual inspection.
            w, row_height, top = 280, 308, 80
            comparison = Image.new('RGB', (3*w+32, top+8*row_height+20), 'white')
            cd = ImageDraw.Draw(comparison)
            cd.text((16, 8), f'Subject 2 / E109 / fixed test IDs {page*8+1}-{page*8+8}', fill='black', font=title_font)
            for col, title in enumerate(['GT', 'K1 (candidate 0)', 'Brain best-of-8']):
                cd.text((16+col*w, 48), title, fill='black', font=font)
            cw, ch, ctop = 200, 238, 80
            candidates = Image.new('RGB', (9*cw+32, ctop+8*ch+20), 'white')
            dd = ImageDraw.Draw(candidates)
            dd.text((16, 8), f'Subject 2 / fixed candidate archive / page {page+1} / green = selected', fill='black', font=title_font)
            for col, title in enumerate(['GT'] + [f'Candidate {j}' for j in range(8)]):
                dd.text((16+col*cw, 48), title, fill='black', font=font)
            for row, i in enumerate(ids[page*8:page*8+8]):
                rec = json.loads((O / 'records' / f'{i}.json').read_text())
                with np.load(O / 'candidates' / f'image_{i:06d}.npz') as a:
                    imgs = a['candidate_images'].copy()
                gt = h5['imgBrick'][i]
                winner = rec['selected_index']
                for col, arr in enumerate([gt, imgs[0], imgs[winner]]):
                    x, y = 16+col*w, top+row*row_height
                    cd.text((x, y), f'ID {i}' if col == 0 else (f'c0' if col == 1 else f'c{winner} / r={rec["correlations"][winner]:.4f}'), fill='black', font=font)
                    comparison.paste(Image.fromarray(arr).resize((264,264), Image.Resampling.LANCZOS), (x,y+26))
                for col, arr in enumerate([gt] + list(imgs)):
                    x, y = 16+col*cw, ctop+row*ch
                    label = f'ID {i}' if col == 0 else f'c{col-1} / r={rec["correlations"][col-1]:.4f}'
                    dd.text((x,y), label, fill='black', font=font)
                    candidates.paste(Image.fromarray(arr).resize((188,188), Image.Resampling.LANCZOS), (x,y+26))
                    if col == winner+1:
                        dd.rectangle((x-2,y+24,x+189,y+215), outline='#07834B', width=3)
            for kind, canvas in [('comparison', comparison), ('candidates', candidates)]:
                path = target / f'{kind}-{page+1}.png'
                canvas.save(path)
                manifest['pages'].append(dict(file=path.name, ids=ids[page*8:page*8+8], sha256=sha(path), width=canvas.width, height=canvas.height))
    save(O / 'atlas-manifest.json', manifest)
    print('ATLAS_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['audit', 'atlas'])
    globals()[parser.parse_args().mode]()
