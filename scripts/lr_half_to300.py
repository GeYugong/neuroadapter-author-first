"""One continuous full-state L140-to-L300 run; no intermediate evaluation pauses."""
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
REPORT = ROOT / 'reports/lr-half-to300'

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
    assert source.name == f'checkpoint-{epoch}' and 140 <= epoch < 300
    assert state['run_id'] == RUN.name and state['current_lr'] == [5e-5]
    assert state['global_optimizer_updates'] == epoch * 532 and state['skipped_updates'] == 0
    existing = sorted(int(p.name.split('-')[-1]) for p in RUN.glob('checkpoint-*') if (p/'COMPLETE').exists())
    assert max(existing) == epoch, 'Use the latest complete checkpoint; never replay an earlier epoch'
    assert not list(RUN.glob('checkpoint-*.incomplete')), 'Inspect and retain partial save before explicit recovery'
    free = shutil.disk_usage(ROOT).free
    assert free > 50 * 2**30, 'Insufficient reserve for retained checkpoints and later evaluation'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    cmd = [sys.executable, 'scripts/run_train.py', '--branch-lr-half', '--target', '300', '--resume', str(source)]
    info = dict(source=str(source), completed_epochs=epoch, optimizer_updates=epoch*532,
                target_epochs=300, target_optimizer_updates=159600, learning_rate=5e-5,
                code_commit=commit, command=cmd, disk_free_bytes=free, protected_existing_checkpoints=existing,
                milestones=[200,250,300], controller_pid=os.getpid(),
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
    original = RUN/'checkpoint-140'
    protected = {str(p.relative_to(ROOT)): sha(p) for p in original.iterdir() if p.is_file()}
    l120 = RUN/'exports/epoch-120/inference.pt'
    assert sha(l120) == 'bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798'
    for p in [l120,ROOT/'metric_brain_adapter.py',ROOT/'configs/val_ids.txt',ROOT/'configs/atlas_val32_ids.txt']:
        protected[str(p.relative_to(ROOT))] = sha(p)
    save(REPORT/f'{tag}-protected-before.json',protected)
    logpath = ROOT/f'logs/lr-half-to300-{tag}.log'
    with logpath.open('x') as log:
        save(REPORT/f'{tag}-started.json',info)
        print(json.dumps(info),flush=True)
        result = subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    (ROOT/f'logs/lr-half-to300-{tag}.exit').write_text(str(result.returncode)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    for n in [200,250,300]:
        cp = RUN/f'checkpoint-{n}'
        progress = json.loads((cp/'progress.json').read_text())
        assert (cp/'COMPLETE').exists() and progress['completed_epochs']==n
        assert progress['global_optimizer_updates']==n*532 and progress['skipped_updates']==0
        assert progress['current_lr']==[5e-5] and (RUN/f'exports/epoch-{n}/inference.pt').exists()
    for p,digest in protected.items():
        assert sha(ROOT/p)==digest, p
    save(REPORT/'training-complete.json',dict(exit_code=0,completed_epochs=300,optimizer_updates=159600,
        milestones=[200,250,300],validation_pending=True,code_commit=commit,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('TRAINING_COMPLETE: validation and delivery pending',flush=True)

if __name__ == '__main__':
    main()
