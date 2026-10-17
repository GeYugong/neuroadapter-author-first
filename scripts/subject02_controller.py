"""Run audited Subject 2 preparation, distributed preflight and one fixed training run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/subject02-bf16-seed42'
RUN = ROOT/'runs/subject02-author-bf16-seed42'
PY = ROOT/'runtime/env/bin/python'


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def status(stage, **kw):
    write(OUT/'controller-status.json', dict(stage=stage, controller_pid=os.getpid(), time=time.time(), **kw))


def wait_free():
    while True:
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        memory = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        free = [int(x) for x in memory.splitlines()]
        if not apps and len(free)==2 and min(free)>=22000:
            return
        status('waiting_for_two_free_gpus', free_mib=free)
        time.sleep(30)


def run_stage(name, config, output, target, smoke=False, resume=None):
    start = OUT/(name+'-started.json')
    assert not start.exists(), 'Refusing duplicate stage '+name
    command = [str(PY),'-B','-m','torch.distributed.run','--nproc_per_node=2','--nnodes=1',
               '--master_addr=127.0.0.1','--master_port=29632','--no_python','/bin/bash',
               str(ROOT/'scripts/subject02_rank.sh'),'--config',str(config),'--target',str(target),
               '--output',str(output.relative_to(ROOT))]
    if smoke:
        command.append('--smoke')
    if resume:
        command += ['--resume',str(resume.relative_to(ROOT))]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1', CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(ROOT/'runtime/matplotlib'), HF_HOME=str(ROOT/'runtime/huggingface'))
    with (OUT/(name+'.log')).open('x') as log:
        p = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(start, dict(command=command,pid=p.pid,time=time.time()))
        while p.poll() is None:
            status(name, child_pid=p.pid, output=str(output.relative_to(ROOT)))
            time.sleep(15)
        write(OUT/(name+'-exit.json'),dict(returncode=p.returncode,time=time.time(),pid=p.pid))
    return p.returncode


def exact(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def compare_preflight(continuous, resumed):
    import torch
    names=['pytorch_model.bin','pytorch_model_1.bin','optimizer.bin','custom_checkpoint_0.pkl',
           'random_states_0.pkl','random_states_1.pkl']
    for name in names:
        a=torch.load(continuous/name,map_location='cpu',weights_only=False)
        b=torch.load(resumed/name,map_location='cpu',weights_only=False)
        assert exact(a,b),'Continuous versus resumed mismatch: '+name
        del a,b
    a=torch.load(continuous/'custom_checkpoint_1.pkl',map_location='cpu',weights_only=False)
    b=torch.load(resumed/'custom_checkpoint_1.pkl',map_location='cpu',weights_only=False)
    for field in ['completed_epochs','global_optimizer_updates','actual_samples_processed','skipped_updates',
                  'run_id','current_lr','loader_generator_state']:
        assert exact(a[field],b[field]),field
    assert a['completed_epochs']==2 and a['global_optimizer_updates']==4 and a['skipped_updates']==0
    for directory in (continuous.parent,resumed.parent):
        for rank in (0,1):
            first=json.loads((directory/f'updates-rank{rank}.jsonl').read_text().splitlines()[0])
            assert first['update']==1 and all(x>0 for x in first['update_to_weight_ratio'].values())
            actual=json.loads((directory/f'actual-runtime-rank{rank}.json').read_text())
            assert actual['mixed_precision']=='bf16' and actual['effective_batch']==32
    return dict(exact_model_optimizer_and_rank_rng=True, exact_loader=True, actual_updates_per_smoke=4,
                frozen_gradient_checks_passed=True, first_update_three_groups_nonzero_both_ranks=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'controller-started.json').exists(), 'No generic restart; inspect existing state'
    authorization=json.loads((OUT/'authorization.json').read_text())
    assert authorization['launch_authorized'] and authorization['subject']==2
    for name,known in authorization['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==known,name
    write(OUT/'controller-started.json',dict(pid=os.getpid(),time=time.time(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    while not (OUT/'data-ready.json').exists():
        if (OUT/'data-failure.json').exists():
            raise RuntimeError('Data preparation failed; inspect data-failure.json')
        state=json.loads((OUT/'data-status.json').read_text())
        status('waiting_for_subject02_data',data_state=state)
        if subprocess.run(['tmux','has-session','-t','author-subject02-prepare'],capture_output=True).returncode:
            raise RuntimeError('Preparation exited without data-ready.json')
        time.sleep(30)
    data=json.loads((OUT/'data-ready.json').read_text())
    assert data['subject']==2 and data['train']==8500 and data['selected_vertices_all_finite']
    assert hashlib.sha256((ROOT/'data/subject02/metadata_sub-02.npy').read_bytes()).hexdigest()==data['metadata_sha256']
    selected=None
    for batch,accumulation in [(16,1),(8,2),(4,4)]:
        wait_free()
        config=json.loads((ROOT/'configs/subject02.json').read_text())
        config.update(train_batch_size=batch,gradient_accumulation_steps=accumulation)
        path=OUT/f'preflight-b{batch}-config.json'
        assert not path.exists()
        write(path,config)
        prefix=f'preflight-b{batch}'
        continuous=RUN/'_preflight'/f'b{batch}-continuous'
        resumed=RUN/'_preflight'/f'b{batch}-resumed'
        code=run_stage(prefix+'-continuous',path,continuous,2,smoke=True)
        if code:
            log=(OUT/(prefix+'-continuous.log')).read_text(errors='replace')
            if 'CUDA out of memory' in log or 'torch.OutOfMemoryError' in log:
                # A memory adaptation only; keep failed logs and verify all workers have exited.
                wait_free()
                continue
            raise RuntimeError(prefix+' failed for a non-OOM reason')
        assert run_stage(prefix+'-split1',path,resumed,1,smoke=True)==0
        assert run_stage(prefix+'-resume2',path,resumed,2,smoke=True,resume=resumed/'checkpoint-1')==0
        proof=compare_preflight(continuous/'checkpoint-2',resumed/'checkpoint-2')
        write(OUT/'preflight.json',dict(passed=True,microbatch=batch,accumulation=accumulation,effective_batch=32,
                                       checks=proof,time=time.time(),data_ready_sha256=hashlib.sha256((OUT/'data-ready.json').read_bytes()).hexdigest()))
        selected=path
        break
    assert selected is not None,'No predeclared memory layout passed'
    config=json.loads(selected.read_text())
    write(OUT/'effective-config.json',config)
    wait_free()
    assert not list(RUN.glob('checkpoint-*')) and not (RUN/'exports').exists(), 'Fresh formal initialization required'
    code=run_stage('train-to-500',OUT/'effective-config.json',RUN,500)
    if code:
        raise RuntimeError('Formal training exited nonzero; preserve artifacts and inspect before recovery')
    if (RUN/'intervention-required.json').exists():
        status('stopped_needs_diagnosis',intervention=json.loads((RUN/'intervention-required.json').read_text()))
        return
    checkpoint=RUN/'checkpoint-500'
    assert (checkpoint/'COMPLETE').exists()
    progress=json.loads((checkpoint/'progress.json').read_text())
    assert progress['completed_epochs']==500 and progress['skipped_updates']==0
    status('training_complete_evaluation_pending',checkpoint=str(checkpoint.relative_to(ROOT)),
           updates=progress['global_optimizer_updates'])


if __name__=='__main__':
    try:
        main()
    except Exception:
        value=dict(time=time.time(),traceback=traceback.format_exc())
        write(OUT/'controller-failure.json',value)
        status('failed_needs_diagnosis',error=value)
        raise
