"""Bounded single-variable experiment; preserve all historical artifacts."""
import sys,os,json,time,fcntl,shutil,subprocess,traceback
from pathlib import Path
import balanced_controller as base
ROOT=base.ROOT
REPORT=ROOT/'reports/wd01-from450'
SOURCE=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-from400'
RUN=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-wd01-from450'
base.REPORT=REPORT
read,save,sha,status=base.read,base.save,base.sha,base.status

def main():
    lock=(ROOT/'runtime/wd01-from450.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not RUN.exists(),'Existing branch requires explicit inspection'
    assert shutil.disk_usage(ROOT).free>=50*2**30,'Need 50 GiB including 25 GiB reserve'
    save(REPORT/'controller-started.json',dict(pid=os.getpid(),unix_time=time.time(),maximum_epoch=500))
    protected={str(p.relative_to(ROOT)):sha(p) for p in (SOURCE/'checkpoint-450').iterdir() if p.is_file()}
    tracked=subprocess.check_output(['git','ls-files','-z','reports','benchmarks/subj01-v1','configs','train_brain_adapter.py','metric_brain_adapter.py'],cwd=ROOT).decode().split('\0')
    for rel in tracked:
        if rel and not rel.startswith('reports/wd01-from450/'):protected[rel]=sha(ROOT/rel)
    save(REPORT/'protected-before.json',protected)
    baseline=read(ROOT/'reports/balanced-lr-from400/epoch-450/val500/metrics.json')
    source=SOURCE/'checkpoint-450'
    for target in [475,500]:
        config=read(ROOT/'configs/stage1.json');config.update(run_id=RUN.name,output_dir=str(RUN),learning_rate=1e-5,weight_decay=0.1,num_train_epochs=target,resume_from=str(source),branch_lr_half=True)
        cmd=[sys.executable,'-B','-m','accelerate.commands.launch','--config_file',str(ROOT/'acc_config.yaml'),'--main_process_port','29627','--mixed_precision','no',str(ROOT/'scripts/wd01_train.py')]
        for k,v in config.items():cmd+=['--'+k] if v is True else ['--'+k,str(v)]
        base.execute(f'wd01-train-to-{target}',cmd,True)
        cp=RUN/f'checkpoint-{target}';state=read(cp/'progress.json')
        assert (cp/'COMPLETE').exists() and state['completed_epochs']==target and state['global_optimizer_updates']==target*532 and state['skipped_updates']==0
        assert state['current_lr']==[1e-5] and state['config']['weight_decay']==0.1
        for rank in [0,1]:
            audit=read(RUN/f'branch-resume-{int(source.name.split("-")[-1])}-rank{rank}.json')
            assert all(audit[k] for k in ['model_exact','adamw_before_exact','adamw_after_only_weight_decay_changed','rng_exact','loader_restored','order_matches_historical'])
            assert audit['optimizer_weight_decay']==[0.1]
        ref={x['epoch']:x for x in map(json.loads,(SOURCE/'epochs.jsonl').read_text().splitlines())}
        for x in map(json.loads,(RUN/'epochs.jsonl').read_text().splitlines()):assert x['image_order_sha256']==ref[x['epoch']]['image_order_sha256']
        for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
        save(REPORT/f'checkpoint-{target}-audit.json',dict(progress=state,checkpoint_sha256={p.name:sha(p) for p in cp.iterdir() if p.is_file()},protected_unchanged=True))
        base.probe(cp,f'wd01-probe-{target}-attempt1')
        source=cp
    for phase in ['decode','score']:
        base.execute(f'wd01-val-500-{phase}',[sys.executable,'-B','scripts/wd01_evaluate.py','--phase',phase])
    metrics=read(REPORT/'epoch-500/val500/metrics.json')
    save(REPORT/'gate-500.json',base.gate(metrics,baseline))
    control=read(ROOT/'reports/balanced-lr-from400/epoch-500/val500/metrics.json')
    save(REPORT/'matched-control-deltas.json',dict(delta_S_pp=metrics['S']-control['S'],delta={k:metrics['mean'][k]-v for k,v in control['mean'].items()},reference='balanced-lr-from400/epoch-500'))
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    status('validation_complete_review_pending',execution_complete=False,k4_confirmation_pending=read(REPORT/'gate-500.json')['passed'],maximum_epoch=500)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise
