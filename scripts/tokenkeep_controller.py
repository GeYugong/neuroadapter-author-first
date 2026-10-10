"""Fixed fifty-epoch retention trial; one endpoint, then frozen validation."""
import sys,os,time,json,fcntl,shutil,subprocess,traceback,argparse
from pathlib import Path
import balanced_controller as base
ROOT=base.ROOT
REPORT=ROOT/'reports/tokenkeep-from450'
RUN=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-tokenkeep-from450'
SOURCE=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-from400'
base.REPORT=REPORT
read,save,sha,status=base.read,base.save,base.sha,base.status

def protected_check(protected):
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel

def audit(epoch):
    cp=RUN/f'checkpoint-{epoch}';p=read(cp/'progress.json')
    assert (cp/'COMPLETE').exists() and p['completed_epochs']==epoch and p['global_optimizer_updates']==epoch*532 and p['skipped_updates']==0
    assert p['current_lr']==[1e-5] and p['config']['weight_decay']==1e-6 and p['config']['token_keep_min']==0.5
    ref={x['epoch']:x for x in map(json.loads,(SOURCE/'epochs.jsonl').read_text().splitlines())}
    for x in map(json.loads,(RUN/'epochs.jsonl').read_text().splitlines()):assert x['image_order_sha256']==ref[x['epoch']]['image_order_sha256']
    for rank in [0,1]:
        a=read(RUN/f'branch-resume-{epoch-50}-rank{rank}.json')
        assert all(a[k] for k in ['model_exact','adamw_before_exact','adamw_unchanged','rng_exact','loader_restored','order_matches_historical'])
    save(REPORT/f'checkpoint-{epoch}-audit.json',dict(progress=p,checkpoint_sha256={x.name:sha(x) for x in cp.iterdir() if x.is_file()},export_sha256=sha(RUN/f'exports/epoch-{epoch}/inference.pt')))
    return cp

def startup_audit_if_ready():
    target=REPORT/'startup-audit.json'
    if target.exists():return
    audits=[];first=[]
    for rank in [0,1]:
        ap=RUN/f'branch-resume-450-rank{rank}.json';up=RUN/f'updates-rank{rank}.jsonl'
        if not ap.exists() or not up.exists():return
        raw=up.read_bytes()
        if b'\n' not in raw:return
        a=read(ap);v=json.loads(raw.split(b'\n')[0])
        assert all(a[k] for k in ['model_exact','adamw_before_exact','adamw_unchanged','rng_exact','loader_restored','order_matches_historical'])
        assert v['update']==239401 and not v['skipped'] and all(x>0 for x in v['group_preclip_grad_norm'].values()) and all(x>0 for x in v['update_to_weight_norm'].values())
        audits.append(a);first.append(v)
    save(target,dict(actual_training_started=True,unix_time=time.time(),resume_audits=audits,first_real_updates=first,keep_min=0.5,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))

def execute(tag,cmd,both=False):
    base.wait_gpu(both)
    required_gib=32 if tag in ('tokenkeep-preflight','tokenkeep-train-to-500') else 15
    assert shutil.disk_usage(ROOT).free>=required_gib*2**30, 'Insufficient free disk after resource wait'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1' if both else '0',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'runtime/matplotlib'))
    log=ROOT/'logs'/f'balanced-{tag}.log'
    with log.open('x') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        save(REPORT/f'{tag}-started.json',dict(pid=p.pid,command=cmd,unix_time=time.time()))
        status(tag,child_pid=p.pid,actual_training_started=False if 'train-to' in tag else None)
        while p.poll() is None:
            if 'train-to' in tag:
                startup_audit_if_ready()
                status(tag,child_pid=p.pid,actual_training_started=(REPORT/'startup-audit.json').exists())
            time.sleep(10)
        if 'train-to' in tag:startup_audit_if_ready()
        rc=p.returncode
    save(REPORT/f'{tag}-exit.json',dict(exit_code=rc,unix_time=time.time()))
    assert rc==0,(tag,rc)

def main():
    lock=(ROOT/'runtime/tokenkeep-from450.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(REPORT/'engineering-audit.json')['passed']
    assert not RUN.exists(),'Existing run requires inspected explicit recovery'
    assert shutil.disk_usage(ROOT).free>=32*2**30
    save(REPORT/'controller-started.json',dict(pid=os.getpid(),unix_time=time.time(),maximum_epoch=500,first_gate_epoch=500,actual_training_started=False))
    paths=subprocess.check_output(['git','ls-files','-z','reports','benchmarks/subj01-v1','configs','train_brain_adapter.py','metric_brain_adapter.py'],cwd=ROOT).decode().split('\0')
    protected={p:sha(ROOT/p) for p in paths if p and not p.startswith('reports/tokenkeep-from450/')}
    for p in (SOURCE/'checkpoint-450').iterdir():
        if p.is_file():protected[str(p.relative_to(ROOT))]=sha(p)
    save(REPORT/'protected-before.json',protected)
    execute('tokenkeep-preflight',[sys.executable,'-B','scripts/tokenkeep_preflight.py'],True)
    assert read(REPORT/'preflight.json')['passed']
    source=SOURCE/'checkpoint-450';reference=read(ROOT/'reports/balanced-lr-from400/epoch-450/val500/metrics.json')
    passed=[]
    for epoch in [500]:
        c=read(ROOT/'configs/stage1.json');c.update(run_id=RUN.name,output_dir=str(RUN),learning_rate=1e-5,weight_decay=1e-6,num_train_epochs=epoch,resume_from=str(source),branch_lr_half=True)
        cmd=[sys.executable,'-B','-m','accelerate.commands.launch','--config_file',str(ROOT/'acc_config.yaml'),'--main_process_port','29631','--mixed_precision','no',str(ROOT/'scripts/tokenkeep_train.py')]
        for k,v in c.items():cmd+=['--'+k] if v is True else ['--'+k,str(v)]
        execute(f'tokenkeep-train-to-{epoch}',cmd,True)
        cp=audit(epoch);protected_check(protected)
        tag=f'tokenkeep-probe-{epoch}-attempt1'
        execute(tag,[sys.executable,'-B','scripts/fixed_noise_probe.py','--checkpoint',str(cp),'--output',str(REPORT/tag),'--output-root','tokenkeep-from450','--budget-seconds','7200'])
        assert read(REPORT/tag/'summary.json')['complete']
        for phase in ['decode','score']:
            execute(f'tokenkeep-val-{epoch}-{phase}',[sys.executable,'-B','scripts/tokenkeep_evaluate.py','--epoch',str(epoch),'--phase',phase])
        metrics=read(REPORT/f'epoch-{epoch}/val500/metrics.json')
        control=read(ROOT/'reports/balanced-lr-from400/epoch-500/val500/metrics.json')
        assert metrics['image_ids']==control['image_ids']==reference['image_ids'], 'Validation ID mismatch'
        save(REPORT/'matched-budget-comparison.json',dict(new=metrics['mean'],control=control['mean'],delta={k:metrics['mean'][k]-control['mean'][k] for k in metrics['mean']},delta_S_pp=metrics['S']-control['S'],same_image_ids=metrics['image_ids']==control['image_ids']))
        gate=base.gate(metrics,reference);gate['epoch']=epoch;save(REPORT/f'gate-{epoch}.json',gate)
        if gate['passed']:passed.append((epoch,metrics))
    protected_check(protected)
    finalist=500 if passed else None
    save(REPORT/'k2-selection.json',dict(candidate_epoch=finalist,eligible_epochs=[e for e,m in passed],new_winner_confirmed=False,k4_required=finalist is not None))
    status('validation_complete_review_pending',execution_complete=False,k4_confirmation_pending=finalist is not None,candidate_epoch=finalist,maximum_epoch=500)
if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise
