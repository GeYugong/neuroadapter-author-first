"""Single explicit recovery of the verified epoch420 snapshot, never restart."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from launch_mindbridge_training import B, R, SOURCE, EXPECTED_SOURCE, OUT, CMD


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(B / 'assets/mindbridge/training.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (B / 'logs/mindbridge-training.exit').read_text().strip() == '1'
        assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == EXPECTED_SOURCE
        assert not subprocess.check_output(['git', '-C', str(SOURCE), 'diff', 'HEAD', '--', 'src'], text=True)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        receipt = json.loads((B / 'mindbridge-data-complete.json').read_text())
        assert sha(B / 'mindbridge-data-validation.json') == receipt['validation_sha256']
        snapshot = json.loads((B / 'mindbridge-recovery-snapshot.json').read_text())
        root = R / snapshot['snapshot']
        for row in snapshot['files']:
            assert sha(root / row['path']) == row['sha256'], row['path']
        assert not (B / 'mindbridge-resume-start.json').exists()
        assert not (OUT / 'best-interrupted-epoch420').exists()
        cmd = [str(B / 'mindbridge_resume_train.py') if x == str(B / 'mindbridge_author_train.py') else x for x in CMD] + ['--resume']
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0,1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   WANDB_MODE='disabled', TORCH_EXTENSIONS_DIR=str(B / 'cache/mindbridge-torch-extensions'),
                   OMP_NUM_THREADS='4', MAX_JOBS='4')
        for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
            env.pop(key, None)
        with open(B / 'logs/mindbridge-resume.log', 'x') as log:
            # Retain the incomplete previous state verbatim; repaired best gets a fresh directory.
            (OUT / 'best').rename(OUT / 'best-interrupted-epoch420')
            with open(B / 'mindbridge-resume-start.json', 'x') as f:
                json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               command=cmd, snapshot=str(root), next_epoch_zero_based=421,
                               expected_global_steps=36206, completed_epochs=421,
                               scope='Synchronization and full-state resume only; original training mathematics retained'), f, indent=2)
                f.write('\n')
            rc = subprocess.call(cmd, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT)
        (B / 'logs/mindbridge-resume.exit').write_text(str(rc) + '\n')
        sys.exit(rc)


if __name__ == '__main__':
    main()
