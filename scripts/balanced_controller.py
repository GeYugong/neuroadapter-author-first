"""Bounded, resource-aware continuation. Failures require explicit inspection."""
import json,os,sys,time,subprocess,hashlib,shutil,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/balanced-lr-from400'
OLD=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
RUN=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-from400'
def read(p):return json.loads(p.read_text())
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def status(phase,**kw):
    p=REPORT/'controller-status.tmp';p.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),unix_time=time.time(),**kw),indent=2));p.replace(REPORT/'controller-status.json')
def wait_gpu(both=False):
    while True:
        uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        if not any(u in busy for u in (uuids if both else uuids[:1])):return
        status('waiting_two_gpus' if both else 'waiting_gpu0',busy_gpu_uuids=busy);time.sleep(60)
def execute(tag,cmd,both=False):
    wait_gpu(both)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1' if both else '0',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'runtime/matplotlib'))
    log=ROOT/'logs'/f'balanced-{tag}.log'
    with log.open('x') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        save(REPORT/f'{tag}-started.json',dict(pid=p.pid,command=cmd,unix_time=time.time()))
        status(tag,child_pid=p.pid)
        rc=p.wait()
    save(REPORT/f'{tag}-exit.json',dict(exit_code=rc,unix_time=time.time()))
    assert rc==0,(tag,rc)
def probe(cp,tag):
    execute(tag,[sys.executable,'-B','scripts/fixed_noise_probe.py','--checkpoint',str(cp),'--output',str(REPORT/tag),'--budget-seconds','7200'])
    result=read(REPORT/tag/'summary.json')
    assert result.get('complete') or result.get('budget_hit'),result
def gate(new,old):
    a=new['mean'];b=old['mean'];ds=new['S']-old['S']
    checks=dict(non_degrading=new['S']>=old['S'] and a['PixCorr']>=b['PixCorr'] and a['SSIM']>=b['SSIM'],material=ds>=.3 or (a['PixCorr']-b['PixCorr']>=.005 and a['SSIM']-b['SSIM']>=.005),recognition=all(a[k]>=b[k]-.5 for k in ['Alex(2)','Alex(5)','Incep','CLIP']),distances=all(a[k]<=b[k]+.005 for k in ['Eff','SwAV']))
    return dict(passed=all(checks.values()),checks=checks,delta_S_pp=ds,delta={k:a[k]-b[k] for k in a},budget_rule_not_significance=True)
def main():
    lock=(ROOT/'runtime/balanced-lr-from400.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(REPORT/'controller-started.json',dict(pid=os.getpid(),unix_time=time.time(),maximum_epoch=500))
    # Adopt the already running diagnostic without launching a duplicate.
    exitpath=ROOT/'logs/balanced-probe-old400-attempt1.exit'
    while not exitpath.exists():
        started=read(REPORT/'probe-old400-attempt1/started.json')
        os.kill(started['pid'],0)
        status('probe_old400',child_pid=started['pid']);time.sleep(30)
    assert int(exitpath.read_text())==0
    first=read(REPORT/'probe-old400-attempt1/summary.json')
    assert first.get('complete') or first.get('budget_hit'),first
    probe(OLD/'checkpoint-500','probe-old500-attempt1')
    audits=[read(REPORT/x/'load-audit.json') for x in ['probe-old400-attempt1','probe-old500-attempt1']]
    assert audits[0]['query_manifest_sha256']==audits[1]['query_manifest_sha256']
    for audit in audits:assert all(audit[k] for k in ['strict_complete_model','strict_guidance','parcel_identity']) and audit['new_training_updates']==0
    save(REPORT/'diagnostic-preflight.json',dict(hard_checks_passed=True,query_manifest_sha256=audits[0]['query_manifest_sha256'],new_training_updates=0))
    protected={str(p.relative_to(ROOT)):sha(p) for p in (OLD/'checkpoint-400').iterdir() if p.is_file()}
    tracked=subprocess.check_output(['git','ls-files','-z','reports','benchmarks/subj01-v1','configs','train_brain_adapter.py','metric_brain_adapter.py'],cwd=ROOT).decode().split('\0')
    for rel in tracked:
        if rel and not rel.startswith('reports/balanced-lr-from400/'):protected[rel]=sha(ROOT/rel)
    save(REPORT/'protected-before.json',protected)
    assert not RUN.exists(),'Existing run needs explicit inspection'
    assert shutil.disk_usage(ROOT).free>=70*2**30,'Initial disk budget requires 70 GiB'
    source=OLD/'checkpoint-400'
    baseline=read(ROOT/'reports/lr-half-to500/epoch-400/val500/metrics.json')
    for target in [425,450,475,500]:
        if target>450 and not read(REPORT/'gate-450.json')['passed']:break
        config=read(ROOT/'configs/stage1.json');config.update(run_id=RUN.name,output_dir=str(RUN),learning_rate=1e-5,num_train_epochs=target,resume_from=str(source),branch_lr_half=True)
        cmd=[sys.executable,'-B','-m','accelerate.commands.launch','--config_file',str(ROOT/'acc_config.yaml'),'--main_process_port','29623','--mixed_precision','no',str(ROOT/'scripts/balanced_train.py')]
        for k,v in config.items():cmd+=['--'+k] if v is True else ['--'+k,str(v)]
        execute(f'train-to-{target}',cmd,True)
        cp=RUN/f'checkpoint-{target}';state=read(cp/'progress.json')
        assert (cp/'COMPLETE').exists() and state['completed_epochs']==target and state['global_optimizer_updates']==target*532 and state['skipped_updates']==0 and state['current_lr']==[1e-5]
        for rank in [0,1]:
            audit=read(RUN/f'branch-resume-{int(source.name.split("-")[-1])}-rank{rank}.json')
            assert all(audit[k] for k in ['model_exact','adamw_before_exact','adamw_after_only_lr_changed','rng_exact','loader_restored','order_matches_historical'])
        reference={x['epoch']:x for x in map(json.loads,(OLD/'epochs.jsonl').read_text().splitlines())}
        for x in map(json.loads,(RUN/'epochs.jsonl').read_text().splitlines()):assert x['image_order_sha256']==reference[x['epoch']]['image_order_sha256']
        for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
        save(REPORT/f'checkpoint-{target}-audit.json',dict(progress=state,checkpoint_sha256={p.name:sha(p) for p in cp.iterdir() if p.is_file()},protected_unchanged=True))
        probe(cp,f'probe-new{target}-attempt1')
        if target in [450,500]:
            for phase in ['decode','score']:execute(f'val-{target}-{phase}',[sys.executable,'-B','scripts/balanced_evaluate.py','--epoch',str(target),'--phase',phase])
            result=read(REPORT/f'epoch-{target}/val500/metrics.json')
            save(REPORT/f'gate-{target}.json',dict(epoch=target,**gate(result,baseline)))
        source=cp
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    status('validation_complete_review_pending',final_epoch=state['completed_epochs'],execution_complete=False,standard_test_pending=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise
