"""Bounded spatial-loss trial with a predeclared L475 continuation gate."""
import sys,os,time,json,fcntl,shutil,subprocess,traceback,argparse
from pathlib import Path
import balanced_controller as base
ROOT=base.ROOT
REPORT=ROOT/'reports/spatial-l1-from450'
RUN=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-spatiall1-from450'
SOURCE=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-from400'
base.REPORT=REPORT
read,save,sha,status=base.read,base.save,base.sha,base.status

def protected_check(protected):
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel

def audit(epoch):
    cp=RUN/f'checkpoint-{epoch}';p=read(cp/'progress.json')
    assert (cp/'COMPLETE').exists() and p['completed_epochs']==epoch and p['global_optimizer_updates']==epoch*532 and p['skipped_updates']==0
    assert p['current_lr']==[1e-5] and p['config']['weight_decay']==1e-6
    ref={x['epoch']:x for x in map(json.loads,(SOURCE/'epochs.jsonl').read_text().splitlines())}
    for x in map(json.loads,(RUN/'epochs.jsonl').read_text().splitlines()):assert x['image_order_sha256']==ref[x['epoch']]['image_order_sha256']
    for rank in [0,1]:
        a=read(RUN/f'branch-resume-{epoch-25}-rank{rank}.json')
        assert all(a[k] for k in ['model_exact','adamw_before_exact','adamw_unchanged','rng_exact','loader_restored','order_matches_historical'])
    save(REPORT/f'checkpoint-{epoch}-audit.json',dict(progress=p,checkpoint_sha256={x.name:sha(x) for x in cp.iterdir() if x.is_file()},export_sha256=sha(RUN/f'exports/epoch-{epoch}/inference.pt')))
    return cp

def main():
    lock=(ROOT/'runtime/spatial-l1-from450.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(REPORT/'preflight.json')['passed']
    assert not RUN.exists(),'Existing run requires inspected explicit recovery'
    assert shutil.disk_usage(ROOT).free>=50*2**30
    save(REPORT/'controller-started.json',dict(pid=os.getpid(),unix_time=time.time(),maximum_epoch=500,first_gate_epoch=475))
    paths=subprocess.check_output(['git','ls-files','-z','reports','benchmarks/subj01-v1','configs','train_brain_adapter.py','metric_brain_adapter.py'],cwd=ROOT).decode().split('\0')
    protected={p:sha(ROOT/p) for p in paths if p and not p.startswith('reports/spatial-l1-from450/')}
    for p in (SOURCE/'checkpoint-450').iterdir():
        if p.is_file():protected[str(p.relative_to(ROOT))]=sha(p)
    save(REPORT/'protected-before.json',protected)
    source=SOURCE/'checkpoint-450';reference=read(ROOT/'reports/balanced-lr-from400/epoch-450/val500/metrics.json')
    passed=[]
    for epoch in [475,500]:
        c=read(ROOT/'configs/stage1.json');c.update(run_id=RUN.name,output_dir=str(RUN),learning_rate=1e-5,weight_decay=1e-6,num_train_epochs=epoch,resume_from=str(source),branch_lr_half=True)
        cmd=[sys.executable,'-B','-m','accelerate.commands.launch','--config_file',str(ROOT/'acc_config.yaml'),'--main_process_port','29629','--mixed_precision','no',str(ROOT/'scripts/spatial_train.py')]
        for k,v in c.items():cmd+=['--'+k] if v is True else ['--'+k,str(v)]
        base.execute(f'spatial-train-to-{epoch}',cmd,True)
        cp=audit(epoch);protected_check(protected)
        tag=f'spatial-probe-{epoch}-attempt1'
        base.execute(tag,[sys.executable,'-B','scripts/fixed_noise_probe.py','--checkpoint',str(cp),'--output',str(REPORT/tag),'--output-root','spatial-l1-from450','--budget-seconds','7200'])
        assert read(REPORT/tag/'summary.json')['complete']
        for phase in ['decode','score']:
            base.execute(f'spatial-val-{epoch}-{phase}',[sys.executable,'-B','scripts/spatial_evaluate.py','--epoch',str(epoch),'--phase',phase])
        metrics=read(REPORT/f'epoch-{epoch}/val500/metrics.json')
        gate=base.gate(metrics,reference);gate['epoch']=epoch;save(REPORT/f'gate-{epoch}.json',gate)
        if gate['passed']:passed.append((epoch,metrics))
        if epoch==475 and not gate['passed']:break
        source=cp
    protected_check(protected)
    # Pareto-front candidates over semantic score and the two structure metrics.
    frontier=[]
    for e,m in passed:
        v=[m['S'],m['mean']['PixCorr'],m['mean']['SSIM']]
        dominated=False
        for oe,om in passed:
            ov=[om['S'],om['mean']['PixCorr'],om['mean']['SSIM']]
            if all(x>=y for x,y in zip(ov,v)) and any(x>y for x,y in zip(ov,v)):dominated=True
        if not dominated:frontier.append((e,m))
    finalist=sorted(frontier,key=lambda p:(-p[1]['S'],-p[1]['mean']['PixCorr'],-p[1]['mean']['SSIM'],p[0]))[0][0] if frontier else None
    save(REPORT/'k2-selection.json',dict(candidate_epoch=finalist,eligible_epochs=[e for e,m in passed],new_winner_confirmed=False,k4_required=finalist is not None))
    status('validation_complete_review_pending',execution_complete=False,k4_confirmation_pending=finalist is not None,candidate_epoch=finalist,maximum_epoch=500)
if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise
