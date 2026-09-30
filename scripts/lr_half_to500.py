"""One continuous full-state L300-to-L500 run; no intermediate evaluation pauses."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/subject01-author-fp32-seed42-lr5e-5-from100'
REPORT = ROOT / 'reports/lr-half-to500'

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    source = Path(args.resume).resolve()
    assert source.parent == RUN and (source / 'COMPLETE').exists()
    state = json.loads((source / 'progress.json').read_text())
    epoch = state['completed_epochs']
    assert source.name == f'checkpoint-{epoch}' and 300 <= epoch < 500
    assert state['run_id'] == RUN.name and state['current_lr'] == [5e-5]
    assert state['global_optimizer_updates'] == epoch * 532 and state['skipped_updates'] == 0
    existing = sorted(int(p.name.split('-')[-1]) for p in RUN.glob('checkpoint-*') if (p/'COMPLETE').exists())
    assert max(existing) == epoch, 'Use the latest complete checkpoint; never replay an earlier epoch'
    assert not list(RUN.glob('checkpoint-*.incomplete')), 'Inspect and retain partial save before explicit recovery'
    free = shutil.disk_usage(ROOT).free
    assert free > 50 * 2**30, 'Insufficient reserve for retained checkpoints and later evaluation'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    cmd = [sys.executable, 'scripts/run_train.py', '--branch-lr-half', '--target', '500', '--resume', str(source)]
    info = dict(source=str(source), completed_epochs=epoch, optimizer_updates=epoch*532,
                target_epochs=500, target_optimizer_updates=266000, learning_rate=5e-5,
                code_commit=commit, command=cmd, disk_free_bytes=free, protected_existing_checkpoints=existing,
                milestones=[400,450,500], controller_pid=os.getpid(),
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if not args.launch:
        print(json.dumps(info,indent=2)); return
    lock = (ROOT/'runtime/lr-half.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU busy; do not interrupt other jobs'
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip(), 'Commit tracked engineering changes before launch'
    REPORT.mkdir(parents=True,exist_ok=True)
    tag = f'from-{epoch}'
    assert not (REPORT/f'{tag}-started.json').exists(), 'Prior attempt exists: inspect logs and do not duplicate'
    info['source_files_sha256'] = {p.name: sha(p) for p in source.iterdir() if p.is_file()}
    original = RUN/'checkpoint-300'
    protected = {str(p.relative_to(ROOT)): sha(p) for directory in [original, RUN/'checkpoint-140'] for p in directory.iterdir() if p.is_file()}
    previous = json.loads((ROOT/'reports/lr-half-to300/training-final-audit.json').read_text())['milestones']['300']
    for filename, digest in previous['checkpoint_files_sha256'].items():
        assert sha(original/filename) == digest, filename
    assert sha(RUN/'exports/epoch-300/inference.pt') == previous['export_sha256']
    tracked = subprocess.check_output(['git','ls-files','-z','reports','benchmarks/subj01-v1'],cwd=ROOT).decode().split('\0')
    for rel in tracked:
        if rel and not rel.startswith('reports/lr-half-to500/'):
            protected[rel] = sha(ROOT/rel)
    for export in [120,200,250,300]:
        p = RUN/f'exports/epoch-{export}/inference.pt'
        protected[str(p.relative_to(ROOT))] = sha(p)
    for number in existing:
        for filename in ['COMPLETE','progress.json']:
            p = RUN/f'checkpoint-{number}'/filename
            protected[str(p.relative_to(ROOT))] = sha(p)
    l120 = RUN/'exports/epoch-120/inference.pt'
    assert sha(l120) == 'bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798'
    for p in [l120,ROOT/'metric_brain_adapter.py',ROOT/'configs/val_ids.txt',ROOT/'configs/atlas_val32_ids.txt']:
        protected[str(p.relative_to(ROOT))] = sha(p)
    save(REPORT/f'{tag}-protected-before.json',protected)
    logpath = ROOT/f'logs/lr-half-to500-{tag}.log'
    with logpath.open('x') as log:
        save(REPORT/f'{tag}-started.json',info)
        print(json.dumps(info),flush=True)
        result = subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    (ROOT/f'logs/lr-half-to500-{tag}.exit').write_text(str(result.returncode)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    milestones = {}
    for n in [400,450,500]:
        cp = RUN/f'checkpoint-{n}'
        progress = json.loads((cp/'progress.json').read_text())
        assert (cp/'COMPLETE').exists() and progress['completed_epochs']==n
        assert progress['global_optimizer_updates']==n*532 and progress['skipped_updates']==0
        assert progress['current_lr']==[5e-5] and (RUN/f'exports/epoch-{n}/inference.pt').exists()
        milestones[str(n)] = dict(progress=progress, export_sha256=sha(RUN/f'exports/epoch-{n}/inference.pt'), checkpoint_files_sha256={p.name: sha(p) for p in cp.iterdir() if p.is_file()})
    for p,digest in protected.items():
        assert sha(ROOT/p)==digest, p
    save(REPORT/'training-complete.json',dict(exit_code=0,completed_epochs=500,optimizer_updates=266000,
        milestones=milestones,skipped_updates=0,protected_files_verified=len(protected),validation_pending=True,code_commit=commit,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('TRAINING_COMPLETE: validation and delivery pending',flush=True)

if __name__ == '__main__':
    main()
