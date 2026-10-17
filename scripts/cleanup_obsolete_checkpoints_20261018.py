"""Remove an explicit list of obsolete full checkpoints after preservation checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
import zipfile

ROOT = Path('/data1/matengyu/geyugong/neuroadapter-author-first').resolve()
OUT = ROOT / 'reports/checkpoint-cleanup-20261018'
BASE = 'subject01-author-fp32-seed42'
OLD = BASE + '-lr5e-5-from100'
LOW = BASE + '-lr1e-5-from400'
SPATIAL = BASE + '-lr1e-5-spatiall1-from450'
TOKEN = BASE + '-lr1e-5-tokenkeep-from450'
WD = BASE + '-lr1e-5-wd01-from450'
SEED43 = 'subject01-author-fp32-seed43-lrhalf-at100'
REMOVE = {BASE: [109], OLD: [110, 130, 140, 200, 250, 299, 450, 499],
          LOW: [499], SPATIAL: [474], TOKEN: [499], WD: [499], SEED43: [119]}
KEEP = {BASE: [100, 110], OLD: [120, 300, 400, 500], LOW: [450, 500],
        SPATIAL: [475], TOKEN: [500], WD: [500], SEED43: [100, 120]}
CORE = {'COMPLETE', 'progress.json', 'pytorch_model.bin', 'pytorch_model_1.bin',
        'optimizer.bin', 'custom_checkpoint_0.pkl', 'custom_checkpoint_1.pkl',
        'random_states_0.pkl', 'random_states_1.pkl'}
DELETE_PATHS = [ROOT / 'runs' / run / f'checkpoint-{epoch}'
                for run, epochs in REMOVE.items() for epoch in epochs]
assert len(DELETE_PATHS) == 14


def emit(**data):
    print(json.dumps(data), flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_path(path):
    path = path.absolute()
    path.relative_to(ROOT)
    assert path.resolve() == path and path != ROOT, str(path)
    for p in [path, *path.parents]:
        if p == ROOT:
            break
        assert not p.is_symlink(), str(p)
    return path


def file_metadata(path):
    s = path.lstat()
    return [s.st_size, s.st_mtime_ns, s.st_ino, s.st_mode, s.st_nlink, s.st_uid]


def check_idle():
    blocked = []
    markers = ('train_brain_adapter.py', 'balanced_train.py', 'spatial_train.py',
               'tokenkeep_train.py', 'wd01_train.py', 'lr_half_train.py',
               'balanced_controller.py', 'spatial_controller.py',
               'tokenkeep_controller.py', 'wd01_controller.py')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if any(marker in command for marker in markers):
                blocked.append(int(proc.name))
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if any(target.startswith(str(p) + '/') for p in DELETE_PATHS):
                        blocked.append(int(proc.name))
                except (FileNotFoundError, PermissionError, OSError):
                    pass
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    assert not blocked, f'Relevant processes or open checkpoint readers: {sorted(set(blocked))}'


def checkpoint(path, inspect_archives=False):
    safe_path(path)
    assert path.is_dir() and {p.name for p in path.iterdir()} == CORE, str(path)
    files = []
    for f in sorted(path.iterdir()):
        safe_path(f)
        s = f.lstat()
        assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_size > 0
        if inspect_archives and f.suffix in ('.bin', '.pkl'):
            with zipfile.ZipFile(f) as archive:
                assert any(n.endswith('/data.pkl') for n in archive.namelist()), str(f)
        files.append(dict(path=str(f.relative_to(ROOT)), metadata=file_metadata(f), allocated_bytes=s.st_blocks * 512))
    progress = json.loads((path / 'progress.json').read_text())
    epoch = int(path.name.split('-')[-1])
    assert progress['run_id'] == path.parent.name and progress['completed_epochs'] == epoch
    assert progress['global_optimizer_updates'] == epoch * 532 and progress['skipped_updates'] == 0
    assert (path / 'COMPLETE').read_text().strip() == 'epoch boundary'
    export = path.parent / 'exports' / f'epoch-{epoch}'
    return dict(path=str(path.relative_to(ROOT)), epoch=epoch, progress=progress,
                export_retained=(export / 'inference.pt').is_file(), files=files)


def verify_retained():
    kept = []
    for run, epochs in KEEP.items():
        for epoch in epochs:
            value = checkpoint(ROOT / 'runs' / run / f'checkpoint-{epoch}', True)
            assert value['export_retained']
            kept.append(dict(path=value['path'], epoch=epoch, checkpoint_files=len(value['files']), archives_readable=True))
    expected = json.loads((ROOT / 'reports/balanced-lr-from400/checkpoint-450-audit.json').read_text())['checkpoint_sha256']
    best = ROOT / 'runs' / LOW / 'checkpoint-450'
    hashes = {}
    for name, known in expected.items():
        f = best / name
        current = digest(f)
        assert current == known, str(f)
        hashes[str(f.relative_to(ROOT))] = current
    release = json.loads((ROOT / 'reports/hf-release-20261017/upload-manifest.json').read_text())
    package = next(p for p in release['packages'] if p['repo'] == 'gugabobo/neuroadapter-subject01-l450')
    for f in package['files']:
        current = digest(ROOT / f['source'])
        assert current == f['sha256']
        hashes[f['source']] = current
    emit(event='retained_states_verified', complete_checkpoints=len(kept), l450_sha256_files=len(hashes))
    return kept, hashes


def snapshot_retained():
    """Metadata guard for all existing run, report and benchmark artifacts."""
    skip = set(DELETE_PATHS) | {OUT}
    rows = {}
    for top in ('runs', 'reports', 'benchmarks'):
        for folder, dirs, names in os.walk(ROOT / top, followlinks=False):
            base = Path(folder)
            dirs[:] = sorted(n for n in dirs if base / n not in skip and not (base / n).is_symlink())
            for name in sorted(names):
                p = base / name
                rows[str(p.relative_to(ROOT))] = file_metadata(p)
    raw = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    return rows, hashlib.sha256(raw).hexdigest()


def evidence():
    items = [
        'reports/lr-half-to500/training-complete.json',
        'reports/lr-half-to500/selection.json',
        'reports/balanced-lr-from400/train-to-500-exit.json',
        'reports/balanced-lr-from400/selection.json',
        'reports/spatial-l1-from450/final-audit.json',
        'reports/tokenkeep-from450/final-selection.json',
        'reports/wd01-from450/selection-final.json',
        'reports/seed43-hl-replication/TRAINING_COMPLETE',
    ]
    return {p: digest(ROOT / p) for p in items}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'execute'))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    check_idle()
    retained, best_hashes = verify_retained()
    rows, snapshot_hash = snapshot_retained()
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    targets = []
    for p in DELETE_PATHS:
        value = checkpoint(p)
        assert value['epoch'] < max(KEEP[p.parent.name])
        for f in value['files']:
            assert f['path'] not in tracked and f['metadata'][4] == 1
        value['reason'] = 'Completed run; final and key selected/source full states retained. All existing inference exports and results retained.'
        targets.append(value)
    plan = dict(authorization='User requested further cleanup of unused checkpoints after preserving best weights and important results.',
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                script_sha256=digest(Path(__file__)), targets=targets,
                allocated_bytes=sum(f['allocated_bytes'] for t in targets for f in t['files']),
                retained_complete_checkpoints=retained, l450_sha256=best_hashes,
                retained_file_count=len(rows), retained_metadata_sha256=snapshot_hash,
                completion_evidence_sha256=evidence(),
                removed_epoch_resume_limitation='Deleted full states cannot be resumed at those exact epochs; all inference exports, final full states, selected L450 and key source states are retained.')
    plan_path = OUT / 'plan.json'
    if args.action == 'plan':
        assert not plan_path.exists(), 'Do not overwrite the original cleanup plan'
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
        emit(event='reviewable_plan_ready', checkpoints=len(targets), reclaimable_gib=plan['allocated_bytes']/1024**3,
             retained_checkpoints=len(retained), retained_files=len(rows), targets=[t['path'] for t in targets])
        return
    saved = json.loads(plan_path.read_text())
    for key in ('source_head', 'script_sha256', 'targets', 'retained_complete_checkpoints', 'l450_sha256',
                'retained_file_count', 'retained_metadata_sha256', 'completion_evidence_sha256'):
        assert plan[key] == saved[key], f'Plan changed: {key}'
    assert not (OUT / 'deleted.jsonl').exists() and not (OUT / 'result.json').exists()
    before = shutil.disk_usage(ROOT).free
    for target in targets:
        check_idle()
        p = safe_path(ROOT / target['path'])
        assert checkpoint(p) == {k: v for k, v in target.items() if k != 'reason'}
        shutil.rmtree(p)
        with (OUT / 'deleted.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(path=target['path'], unix_time=time.time())) + '\n')
        emit(event='obsolete_checkpoint_removed', path=target['path'])
    after_rows, after_hash = snapshot_retained()
    assert after_rows == rows and after_hash == snapshot_hash, 'A retained artifact changed'
    kept_after, hashes_after = verify_retained()
    assert kept_after == retained and hashes_after == best_hashes
    assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT).strip()
    result = dict(completed=True, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  deleted_checkpoints=len(targets), deleted_paths=[t['path'] for t in targets],
                  freed_allocated_bytes=plan['allocated_bytes'], free_before_bytes=before,
                  free_after_bytes=shutil.disk_usage(ROOT).free,
                  retained_file_count=len(rows), retained_metadata_sha256=after_hash,
                  all_retained_file_metadata_unchanged=True, l450_checkpoint_and_export_sha256_unchanged=True,
                  retained_complete_checkpoints=len(retained), all_existing_inference_exports_retained=True,
                  other_users_files_touched=False, training_or_evaluation_started=False)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    emit(**result)


if __name__ == '__main__':
    main()
