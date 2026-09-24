"""Launch the patched author entry point; no scientific implementation here."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, required=True)
parser.add_argument('--run-dir')
parser.add_argument('--resume')
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--stage2', action='store_true')
parser.add_argument('--stage3', action='store_true')
parser.add_argument('--stage4', action='store_true')
parser.add_argument('--stage5', action='store_true')
parser.add_argument('--stage6', action='store_true')
parser.add_argument('--branch-lr-half', action='store_true')
parser.add_argument('--seed43-replication', action='store_true')
args = parser.parse_args()
assert sum([args.stage2,args.stage3,args.stage4,args.stage5,args.stage6,args.branch_lr_half,args.seed43_replication]) <= 1
if args.seed43_replication:
    assert args.target in (100,120) and not args.smoke and not args.run_dir
    assert args.target==100 or args.resume
elif args.branch_lr_half:
    assert args.target in (110,120,130,140) and args.resume and not args.smoke and not args.run_dir
elif args.stage6:
    assert args.target in (110,120) and args.resume and not args.smoke and not args.run_dir
    checkpoint=root/args.resume
    assert (checkpoint/'COMPLETE').exists()
    state=json.loads((checkpoint/'progress.json').read_text())
    assert args.target-10 <= state['completed_epochs'] < args.target
    assert state['global_optimizer_updates'] == state['completed_epochs']*532
elif args.stage5:
    assert args.target in (90,100) and args.resume and not args.smoke and not args.run_dir
    checkpoint=root/args.resume
    assert (checkpoint/'COMPLETE').exists()
    state=json.loads((checkpoint/'progress.json').read_text())
    assert args.target-10 <= state['completed_epochs'] < args.target
    assert state['global_optimizer_updates'] == state['completed_epochs']*532
elif args.stage4:
    assert args.target in (70,80) and args.resume and not args.smoke and not args.run_dir
    checkpoint=root/args.resume
    assert (checkpoint/'COMPLETE').exists()
    state=json.loads((checkpoint/'progress.json').read_text())
    assert args.target-10 <= state['completed_epochs'] < args.target
    assert state['global_optimizer_updates'] == state['completed_epochs']*532
elif args.stage3:
    assert args.target == 60 and args.resume and not args.smoke and not args.run_dir
    checkpoint=root/args.resume
    assert (checkpoint/'COMPLETE').exists()
    state=json.loads((checkpoint/'progress.json').read_text())
    assert 50 <= state['completed_epochs'] < 60
    assert state['global_optimizer_updates'] == state['completed_epochs']*532
elif args.stage2:
    assert args.target in (30,40,50) and args.resume and not args.smoke and not args.run_dir
    state=json.loads((Path(args.resume)/'progress.json').read_text())
    assert 20 <= state['completed_epochs'] < args.target
else:
    assert 0 < args.target <= 20
config = json.loads((root/'configs/stage1.json').read_text())
if args.seed43_replication:
    config = json.loads((root/'configs/seed43-replication.json').read_text())
    config['learning_rate'] = 1e-4 if args.target==100 else 5e-5
if args.run_dir:
    config['output_dir'] = str(root/args.run_dir)
if args.smoke:
    config['run_id'] = 'isolated-smoke'
    config['train_ids'] = str(root/'configs/smoke16_ids.txt')
    config['audit'] = True
if args.resume:
    config['resume_from'] = str(root/args.resume)
if args.branch_lr_half:
    config.update(run_id='subject01-author-fp32-seed42-lr5e-5-from100',output_dir=str(root/'runs/subject01-author-fp32-seed42-lr5e-5-from100'),learning_rate=5e-5,branch_lr_half=True)
config['num_train_epochs'] = args.target
if args.stage2: config['stage2'] = True
if args.stage3: config['stage3'] = True
if args.stage4: config['stage4'] = True
if args.stage5: config['stage5'] = True
if args.stage6: config['stage6'] = True
cmd = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file', str(root/'acc_config.yaml'),
       '--main_process_port','29621','--mixed_precision','no',str(root/'train_brain_adapter.py')]
for k,v in config.items():
    cmd += ['--'+k] if v is True else ['--'+k,str(v)]
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0,1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
           OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
           MPLCONFIGDIR=str(root/'runtime/matplotlib'))
print(json.dumps({'command':cmd,'pid':os.getpid()}),flush=True)
result = subprocess.run(cmd,cwd=root,env=env)
print(json.dumps({'exit_code':result.returncode}),flush=True)
sys.exit(result.returncode)
