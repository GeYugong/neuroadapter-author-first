"""Download pinned official Subject1 shards and unpack only author loader inputs.

CPU/network only. No training, model selection, or test scoring.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import fcntl

import numpy as np
import download_p1_models as downloader

B = Path(__file__).resolve().parent
ASSETS = B / 'assets/mindbridge'
DATA = ASSETS / 'data'
LOCK = B / 'mindbridge-data-lock.json'


def unpack(entry):
    path = ASSETS / entry['destination']
    split = Path(entry['destination']).parent.name
    out = DATA / 'webdataset_avg_split' / split / 'subj01'
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            name = member.name
            if not any(name.endswith('.' + ext) for ext in ('jpg', 'nsdgeneral.npy', 'coco73k.npy')):
                continue
            if not member.isfile() or Path(name).name != name:
                raise RuntimeError(f'Unexpected member: {name}')
            dst = out / name
            with archive.extractfile(member) as src:
                content = src.read()
            if dst.exists():
                if dst.read_bytes() != content:
                    raise RuntimeError(f'Existing file mismatch: {dst}')
            else:
                temp = dst.with_suffix(dst.suffix + '.tmp')
                temp.write_bytes(content)
                temp.replace(dst)
            count += 1
    print(json.dumps({'event': 'unpacked', 'shard': entry['name'], 'files': count}), flush=True)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    guard = (ASSETS / '.prepare.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = json.loads(LOCK.read_text())
    if (ASSETS / 'DATA_COMPLETE.json').exists():
        print('Data already prepared; no repeat download/extraction.', flush=True)
        return
    if shutil.disk_usage(B).free < lock['total_bytes'] + 20 * 1024**3:
        raise RuntimeError('Insufficient free disk for pinned data and preparation reserve')
    downloader.A = ASSETS
    downloader.save(ASSETS / 'STARTED.json', {
        'utc': downloader.now(), 'pid': os.getpid(), 'lock_sha256': downloader.sha(LOCK),
        'workers': 2, 'training_started': False, 'total_bytes': lock['total_bytes'],
    })
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(downloader.fetch, lock['files']))
    downloader.save(ASSETS / 'DOWNLOAD_COMPLETE.json', {'utc': downloader.now(), 'files': records})
    for entry in lock['files']:
        unpack(entry)
    test_ids = set(map(int, (B / 'test_full_ids.txt').read_text().split()))
    ids_by_split = {}
    manifests = {}
    for split, expected in [('train', lock['expected_train']), ('val', lock['expected_val'])]:
        root = DATA / 'webdataset_avg_split' / split / 'subj01'
        id_files = sorted(root.glob('*.coco73k.npy'))
        assert len(id_files) == expected, (split, len(id_files), expected)
        rows = []
        for file in id_files:
            prefix = file.name.removesuffix('.coco73k.npy')
            image_id = int(np.load(file, allow_pickle=False).reshape(-1)[0])
            brain_file = root / (prefix + '.nsdgeneral.npy')
            brain = np.load(brain_file, allow_pickle=False)
            assert brain.shape == (3, 15724), (file.name, brain.shape)
            assert np.isfinite(brain).all(), file.name
            image_file = root / (prefix + '.jpg')
            assert image_file.is_file()
            rows.append({'sample': prefix, 'nsd_id0': image_id,
                         'brain_shape': list(brain.shape), 'brain_dtype': str(brain.dtype),
                         'brain_sha256': downloader.sha(brain_file),
                         'image_sha256': downloader.sha(image_file)})
        ids = [r['nsd_id0'] for r in rows]
        assert len(set(ids)) == expected
        assert not set(ids) & test_ids, f'{split} overlaps locked standard test'
        ids_by_split[split] = set(ids)
        manifests[split] = rows
    assert not ids_by_split['train'] & ids_by_split['val']
    # Read-only reuse of the same official NSDAccess metadata and COCO2017 captions.
    links = {
        'nsddata/experiments/nsd/nsd_stim_info_merged.csv': Path('/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd/experiments/nsd/nsd_stim_info_merged.csv'),
        'nsddata_stimuli/stimuli/nsd/annotations/captions_train2017.json': B / 'assets/coco-official/captions_train2017.json',
        'nsddata_stimuli/stimuli/nsd/annotations/captions_val2017.json': B / 'assets/coco-official/captions_val2017.json',
    }
    for relative, source in links.items():
        assert source.is_file(), source
        target = DATA / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.resolve() == source.resolve()
        else:
            target.symlink_to(source)
    downloader.save(B / 'mindbridge-data-validation.json', {
        'utc': downloader.now(), 'source_revision': lock['revision'], 'splits': manifests,
        'train_val_disjoint': True, 'standard_test_disjoint': True,
        'preprocessing': 'Official stored arrays and JPEG bytes; no conversion; author loader does max pooling8192.',
    })
    downloader.save(ASSETS / 'DATA_COMPLETE.json', {
        'utc': downloader.now(), 'n_train': len(ids_by_split['train']), 'n_val': len(ids_by_split['val']),
        'validation_sha256': downloader.sha(B / 'mindbridge-data-validation.json'),
        'new_training_updates': 0,
    })


if __name__ == '__main__':
    main()
