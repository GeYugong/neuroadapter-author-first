"""Prepare independent NSD Subject 2 data without modifying shared assets."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import xml.etree.ElementTree as ET

import h5py
import nibabel as nib
import numpy as np
import requests
import scipy.io as sio
import torch

ROOT = Path('/data1/matengyu/geyugong/neuroadapter-author-first')
OUT = ROOT / 'reports/subject02-bf16-seed42'
DATA = ROOT / 'data/subject02'
RAW = DATA / 'raw'
SHARED = ROOT.parent / 'neuroadapter-subject1-research'
PREFIX = 'nsddata_betas/ppdata/subj02/fsaverage/betas_fithrf_GLMdenoise_RR/'
BASE = 'https://natural-scenes-dataset.s3.amazonaws.com/'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(item):
    path = RAW / Path(item['key']).name
    receipt = OUT / 'downloads' / (path.name + '.json')
    if path.exists():
        old = json.loads(receipt.read_text())
        assert path.stat().st_size == item['size'] and sha(path) == old['sha256']
        assert old['etag'] == item['etag']
        return old
    part = path.with_name(path.name + '.part')
    session = requests.Session()
    session.trust_env = False
    for attempt in range(5):
        try:
            assert shutil.disk_usage(ROOT).free > 40 * 1024**3, 'Less than 40 GiB reserve'
            offset = part.stat().st_size if part.exists() else 0
            assert offset <= item['size']
            if offset < item['size']:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with session.get(BASE + item['key'], headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    assert response.headers['ETag'].strip('"') == item['etag']
                    assert response.status_code == (206 if offset else 200)
                    if offset:
                        assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                    with part.open('ab' if offset else 'wb') as f:
                        for chunk in response.iter_content(4 * 1024**2):
                            f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
            assert part.stat().st_size == item['size']
            value = dict(item, sha256=sha(part), path=str(path.relative_to(ROOT)))
            write(receipt, value)
            part.replace(path)
            return value
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'data-ready.json').exists(), 'Dataset already finalized'
    import fcntl
    lock = (OUT / 'prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(OUT / 'data-status.json', dict(state='downloading', pid=os.getpid(), started=time.time()))
    session = requests.Session()
    session.trust_env = False
    response = session.get(BASE, params={'list-type': '2', 'prefix': PREFIX}, timeout=60)
    response.raise_for_status()
    tree = ET.fromstring(response.content)
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    assert tree.find('s:IsTruncated', ns).text == 'false'
    expected = {f'{h}.betas_session{s:02}.mgh' for h in ('lh', 'rh') for s in range(1, 41)}
    expected |= {'lh.ncsnr.mgh', 'rh.ncsnr.mgh'}
    items = []
    for row in tree.findall('s:Contents', ns):
        key = row.find('s:Key', ns).text
        if Path(key).name in expected:
            items.append(dict(key=key, size=int(row.find('s:Size', ns).text), etag=row.find('s:ETag', ns).text.strip('"')))
    assert {Path(x['key']).name for x in items} == expected and len(items) == 82
    write(OUT / 'download-inventory.json', dict(source=BASE, files=items, bytes=sum(x['size'] for x in items)))
    done = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch, i) for i in items]):
            done.append(future.result())
            write(OUT / 'data-status.json', dict(state='downloading', pid=os.getpid(), completed=len(done), total=82,
                                                downloaded_bytes=sum(x['size'] for x in done), updated=time.time()))
            print(json.dumps(dict(downloaded=len(done), total=82)), flush=True)
    design_path = SHARED / 'data/raw/nsd/experiments/nsd/nsd_expdesign.mat'
    design = sio.loadmat(design_path)
    subject_images = design['subjectim'][1].astype(np.int64) - 1
    order = subject_images[design['masterordering'].reshape(-1).astype(np.int64) - 1]
    shared = np.sort(design['sharedix'].reshape(-1).astype(np.int64) - 1)
    unique, counts = np.unique(order, return_counts=True)
    assert len(order) == 30000 and len(unique) == 10000 and (counts == 3).all()
    pool = np.setdiff1d(subject_images, shared)
    assert len(pool) == 9000 and len(shared) == 1000
    val = np.sort(np.random.default_rng(42).choice(pool, size=500, replace=False))
    train = np.setdiff1d(pool, val)
    metadata = dict(img_presentation_order=order, train_img_num=train, val_img_num=val, test_img_num=shared)
    selected = {}
    parcel_info = {}
    for hemi in ('lh', 'rh'):
        snr = np.asarray(nib.load(str(RAW / f'{hemi}.ncsnr.mgh')).get_fdata(dtype=np.float32)).reshape(-1)
        assert snr.shape == (163842,) and np.isfinite(snr).all()
        metadata[f'{hemi}_ncsnr'] = snr
        pp = DATA / 'parcels' / f'{hemi}_labels_s02.pt'
        labels = torch.load(pp, map_location='cpu', weights_only=True)
        assert len(labels) == 501
        assert torch.equal(torch.cat(labels).sort().values, torch.arange(163842))
        parcels = labels[1:]
        indices = np.argsort([snr[p].mean() for p in parcels])[::-1][:100]
        selected[hemi] = np.unique(np.concatenate([parcels[i].numpy() for i in indices]))
        parcel_info[hemi] = dict(indices=indices.tolist(), max_voxels=max(len(parcels[i]) for i in indices),
                                  selected_vertices=len(selected[hemi]), sha256=sha(pp))
    np.save(DATA / 'metadata_sub-02.npy', metadata, allow_pickle=True)
    for name, ids in [('train', train), ('val', val), ('test', shared), ('smoke', train[:64]), ('probe', val[:32])]:
        np.savetxt(DATA / f'{name}_ids.txt', ids, fmt='%d')
    target = DATA / 'betas_sub-02.h5'
    assert not target.exists()
    partial = target.with_suffix('.h5.partial')
    anomalies = []
    with h5py.File(partial, 'a') as h5:
        h5.attrs.update(subject=2, surface_space='fsaverage', beta_derivative='betas_fithrf_GLMdenoise_RR')
        for hemi in ('lh', 'rh'):
            key = f'{hemi}_betas'
            ds = h5.require_dataset(key, shape=(30000, 163842), dtype='float32', chunks=(8, 163842), compression='lzf')
            completed = int(ds.attrs.get('completed_sessions', 0))
            for number in range(1, 41):
                if number <= completed:
                    continue
                assert shutil.disk_usage(ROOT).free > 35 * 1024**3
                path = RAW / f'{hemi}.betas_session{number:02}.mgh'
                source = nib.load(str(path)).get_fdata(dtype=np.float32)[:, 0, 0, :].T
                assert source.shape == (750, 163842)
                assert np.isfinite(source[:, selected[hemi]]).all(), f'Nonfinite selected vertices: {path}'
                count = int((~np.isfinite(source)).sum())
                if count:
                    anomalies.append(dict(hemisphere=hemi, session=number, nonfinite=count))
                start = (number - 1) * 750
                ds[start:start + 750] = source
                h5.flush()
                for local_row in (0, 375, 749):
                    assert np.array_equal(ds[start + local_row], source[local_row], equal_nan=True)
                ds.attrs['completed_sessions'] = number
                h5.flush()
                write(OUT / 'data-status.json', dict(state='building_hdf5', hemisphere=hemi, session=number, updated=time.time()))
            # Scan every selected value again, including previously committed sessions.
            for start in range(0, 30000, 250):
                assert np.isfinite(ds[start:start + 250][:, selected[hemi]]).all()
    partial.replace(target)
    with h5py.File(SHARED / 'data/raw/nsd/stimuli/nsd_stimuli.hdf5', 'r') as h5:
        assert h5['imgBrick'].shape == (73000, 425, 425, 3)
    ready = dict(subject=2, state='data_ready', train=8500, validation=500, test_excluded_from_training=1000,
                 actual_test_presented=int(np.isin(shared, order).sum()), presentations=30000, repetitions=3,
                 split_seed=42, split_policy='Subject2 nonshared 9000; fixed 500 validation before training',
                 no_added_brain_normalization=True, source_values_preserved=True,
                 selected_vertices_all_finite=True, nonfinite_outside_selected=anomalies, parcels=parcel_info,
                 metadata_sha256=sha(DATA / 'metadata_sub-02.npy'), beta_sha256=sha(target), beta_bytes=target.stat().st_size,
                 design_sha256=sha(design_path), source_files=len(done), completed=time.time())
    write(OUT / 'data-ready.json', ready)
    write(OUT / 'data-status.json', ready)
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(OUT / 'data-failure.json', dict(pid=os.getpid(), time=time.time(), traceback=traceback.format_exc()))
        raise
