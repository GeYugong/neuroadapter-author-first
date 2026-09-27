"""Guard a single original two-GPU Subject1 training run; never evict jobs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

B = Path(__file__).resolve().parent
R = B.parents[1]
SOURCE = B / 'upstream/mindbridge'
EXPECTED_SOURCE = '2a3943ff5b619db6f85f16a5e7ff1331d25def55'
MODEL_NAME = 'benchmark-subj01-vanilla-seed42'
OUT = SOURCE / 'train_logs' / MODEL_NAME
PY = B / 'envs/mindbridge/bin/python'
CMD = [str(PY), '-m', 'accelerate.commands.launch',
       '--config_file', str(B / 'mindbridge-accelerate.yaml'), '--multi_gpu',
       '--num_processes', '2', '--gpu_ids', '0,1', '--main_process_port', '29501',
       '--mixed_precision', 'no', str(B / 'mindbridge_author_train.py'),
       '--model_name', MODEL_NAME, '--subj_list', '1',
       '--data_path', str(B / 'assets/mindbridge/data'),
       '--num_epochs', '600', '--batch_size', '50', '--val_batch_size', '50',
       '--h_size', '2048', '--n_blocks', '4', '--pool_type', 'max', '--pool_num', '8192',
       '--mse_mult', '10000', '--eval_interval', '10', '--ckpt_interval', '10',
       '--max_lr', '3e-4', '--num_workers', '4', '--no-wandb_log']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == EXPECTED_SOURCE
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'diff', 'HEAD', '--', 'src', 'scripts/train_single.sh'], text=True)
    receipt = json.loads((B / 'mindbridge-data-complete.json').read_text())
    assert (receipt['n_train'], receipt['n_val']) == (8559, 300)
    assert sha(B / 'mindbridge-data-validation.json') == receipt['validation_sha256']
    validation = json.loads((B / 'mindbridge-data-validation.json').read_text())
    assert validation['train_val_disjoint'] and validation['standard_test_disjoint']
    assert (B / 'logs/mindbridge-env-install.exit').read_text().strip() == '0'
    assets = json.loads((B / 'mindbridge-model-cpu-check.json').read_text())
    assert assets['full_author_Clipper_loaded'] and assets['CLIP_frozen']
    assert not OUT.exists() or not any(OUT.iterdir()), 'Existing training state: inspect and resume explicitly; never restart.'
    return {'source_commit': EXPECTED_SOURCE, 'command': CMD,
            'data_validation_sha256': receipt['validation_sha256'],
            'output_directory': str(OUT), 'epochs': 600,
            'configuration': 'Original train_single.sh two-rank recipe; local paths and disabled external logging only.',
            'training_started': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    info = preflight()
    if args.check_only:
        print(json.dumps(info, indent=2))
        return
    # Hold the process lock until all launcher children have returned.
    with open(B / 'assets/mindbridge/training.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
        assert len(names) == 2 and all('4090' in n for n in names), names
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        assert not busy, 'GPU compute jobs still active; no training started: ' + busy
        preflight()
        log_path = B / 'logs/mindbridge-training.log'
        start_path = B / 'mindbridge-training-start.json'
        assert not start_path.exists(), 'A previous launch exists; inspect it, do not silently restart.'
        with open(log_path, 'x') as log:
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='0,1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       WANDB_MODE='disabled', TORCH_EXTENSIONS_DIR=str(B / 'cache/mindbridge-torch-extensions'),
                       OMP_NUM_THREADS='4', MAX_JOBS='4')
            for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
                env.pop(key, None)
            info.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), training_started=True,
                        meaning='Launcher dispatched, not proof of completed optimizer updates.')
            with open(start_path, 'x') as f:
                json.dump(info, f, indent=2)
                f.write('\n')
            rc = subprocess.call(CMD, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT)
        (B / 'logs/mindbridge-training.exit').write_text(str(rc) + '\n')
        sys.exit(rc)


if __name__ == '__main__':
    main()
