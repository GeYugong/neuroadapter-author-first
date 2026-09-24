"""Evaluation-only L130 full500 completion and paired L120/L130/L140 comparison."""
import argparse, fcntl, json, sys, time, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import lr_half_evaluate as old
import evaluate_stage as ev
import stage3 as common

ROOT=ev.ROOT; RUN=old.L
REPORT=ROOT/'reports/l130-val500-followup'
WORK=RUN/'comparison/l130-val500-followup'
BASES={120:ROOT/'reports/lr-half-to120',130:REPORT,140:ROOT/'reports/lr-half-to140/epoch-140'}
KEYS=['PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV']
SEED=20260924

def check_protected():
    for path,digest in old.read(REPORT/'protected-before.json').items():
        assert old.sha(ROOT/path)==digest, 'Historical file changed: '+path

def prepare():
    assert not WORK.exists(), 'Existing supplement: inspect before recovery'
    full=old.ids('val_ids.txt'); monitor=old.ids('monitor100_ids.txt')
    assert len(full)==len(set(full))==500 and len(monitor)==len(set(monitor))==100 and set(monitor)<set(full)
    assert (RUN/'checkpoint-140/COMPLETE').exists()
    assert old.read(RUN/'checkpoint-140/progress.json')['global_optimizer_updates']==74480
    provenance={}
    for epoch in (120,130,140):
        folder=RUN/f'decoded/epoch-{epoch}/val'
        ident=old.identity(RUN,epoch,epoch*532,folder)
        auditbase=ROOT/'reports/lr-half-to140/epoch-130' if epoch==130 else BASES[epoch]
        audit=old.read(auditbase/'delivery-checks.json')
        assert audit['export_sha256']==ident['sha256']
        expected=monitor if epoch==130 else full
        assert {int(f.stem.split('_')[1]) for f in folder.glob('image_*.npz')}==set(expected)
        filehash={}
        for i in expected:
            f=folder/f'image_{i:06d}.npz'
            old.verify_file(f,i,RUN/f'decoded/epoch-120/val/{f.name}')
            filehash[str(i)]=old.sha(f)
        provenance[str(epoch)]={'identity':ident,'outputs_sha256':filehash}
    # Freeze all prior reports, mainline/L progress, relevant full states/exports and epoch logs.
    protected=set()
    for f in (ROOT/'reports').rglob('*'):
        if f.is_file() and REPORT not in f.parents: protected.add(f)
    for run in (RUN,old.MAIN):
        protected.update(run.glob('checkpoint-*/progress.json'))
        protected.update(run.glob('exports/epoch-*/inference.pt'))
        protected.update(run.glob('best_so_far/*'))
        if (run/'epochs.jsonl').exists(): protected.add(run/'epochs.jsonl')
    for epoch in (110,120,130,140):
        protected.update((RUN/f'checkpoint-{epoch}').glob('*'))
        protected.update((RUN/f'exports/epoch-{epoch}').glob('*'))
    for epoch,info in provenance.items():
        protected.update(RUN/f'decoded/epoch-{epoch}/val/image_{int(i):06d}.npz' for i in info['outputs_sha256'])
    before={str(f.relative_to(ROOT)):old.sha(f) for f in sorted(protected) if f.is_file()}
    WORK.mkdir(parents=True)
    missing=[i for i in full if i not in set(monitor)]
    np.savetxt(WORK/'missing400_ids.txt',missing,fmt='%d')
    np.savetxt(REPORT/'image_ids.txt',full,fmt='%d')
    old.save(REPORT/'protected-before.json',before)
    old.save(REPORT/'provenance.json',dict(epochs=provenance,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),new_image_ids=missing,new_images=400,new_candidate_images=800,training_updates_added=0))

def generate():
    assert not (WORK/'missing400').exists(), 'Do not overwrite or blindly repeat partial generation'
    ev.decode(RUN,130,'val',WORK/'missing400_ids.txt',WORK/'missing400')
    old.identity(RUN,130,69160,WORK/'missing400')
    full=old.ids('val_ids.txt');monitor=set(old.ids('monitor100_ids.txt'))
    union=WORK/'full500';union.mkdir()
    manifest=[]
    for i in full:
        source=(RUN/'decoded/epoch-130/val' if i in monitor else WORK/'missing400')/f'image_{i:06d}.npz'
        old.verify_file(source,i,RUN/f'decoded/epoch-120/val/{source.name}')
        (union/source.name).symlink_to(source)
        manifest.append(dict(image_id=i,source=str(source),sha256=old.sha(source),reused=i in monitor))
    assert len(list((WORK/'missing400').glob('image_*.npz')))==400
    old.save(REPORT/'full500-manifest.json',manifest)

def score():
    ev.metrics(WORK/'full500',ROOT/'configs/val_ids.txt',REPORT/'val500')
    # The same original100 images retain their original100 negative pool.
    ev.metrics(WORK/'full500',ROOT/'configs/monitor100_ids.txt',REPORT/'val100')
    new=old.read(REPORT/'val100/metrics.json')
    previous=old.read(ROOT/'reports/lr-half-to140/epoch-130/val100/metrics.json')
    for k in KEYS: assert np.isclose(new['mean'][k],previous['mean'][k],atol=1e-10,rtol=0),k

def compare():
    full=old.ids('val_ids.txt');frames={};summaries={}
    for epoch,base in BASES.items():
        cs=[pd.read_csv(base/f'val500/candidate-{k}.csv').set_index('image_id') for k in range(2)]
        for f in cs: assert len(f)==500 and f.index.is_unique and set(f.index)==set(full)
        frames[epoch]=(cs[0].loc[full,KEYS]+cs[1].loc[full,KEYS])/2
        frames[epoch]['S']=frames[epoch][['Alex(5)','Incep','CLIP']].mean(axis=1)
        met=old.read(base/'val500/metrics.json')
        assert np.allclose(frames[epoch].mean(),[met['mean'][k] for k in KEYS]+[met['S']],atol=1e-10,rtol=0)
        summaries[str(epoch)]={pool:old.read(base/pool/'metrics.json') for pool in ('val500','val100')}
    rng=np.random.default_rng(SEED);draws=rng.integers(0,500,size=(10000,500));contrasts={}
    for left,right,primary in [(130,120,True),(130,140,True),(140,120,False)]:
        name=f'L{left}-minus-L{right}'
        d=frames[left]-frames[right];values=d.to_numpy();assert np.isfinite(values).all()
        paired=pd.concat([frames[left].add_prefix(f'L{left}_'),frames[right].add_prefix(f'L{right}_'),d.add_prefix('delta_')],axis=1)
        paired.to_csv(REPORT/f'paired-{name}.csv')
        samples=np.empty((10000,9))
        for start in range(0,10000,100):samples[start:start+100]=values[draws[start:start+100]].mean(axis=1)
        rows=[]
        for j,k in enumerate(KEYS+['S']):
            level=.975 if primary and k=='S' else .95
            lo,hi=np.quantile(samples[:,j],[(1-level)/2,1-(1-level)/2])
            rows.append(dict(metric=k,delta=float(values[:,j].mean()),low=float(lo),high=float(hi),interval_level=level,better='lower' if k in ('Eff','SwAV') else 'higher'))
        contrasts[name]=dict(primary=primary,results=rows)
    check_protected()
    old.save(REPORT/'comparison.json',dict(summaries=summaries,contrasts=contrasts,seed=SEED,resamples=10000,
        method='Image-level paired percentile bootstrap after averaging candidates; per-image identification values retain full500 negative pool.',
        limitation='Exploratory reused internal development set with prior L120/L140 results observed. Nominal intervals do not correct prior adaptive training/model selection; not independent-test or cross-training-seed evidence.'))
    old.save(REPORT/'completion.json',dict(training_updates_added=0,new_images=400,new_candidate_images=800,original100_unchanged=True,historical_outputs_and_reports_unchanged=True,primary_contrasts=['L130-minus-L120','L130-minus-L140']))
    (REPORT/'EVALUATION_COMPLETE').write_text('Metrics and paired comparison complete; Chinese report and publication pending.\n')

def event(**kw):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    with (ROOT/'logs/l130-val500-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','generate','score','compare']);args=parser.parse_args()
    if args.phase:
        globals()[args.phase]()
        (REPORT/f'{args.phase}.done').write_text('exit0\n')
        return
    lock=(ROOT/'runtime/l130-val500.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (REPORT/'PLAN.md').exists()
    common.event=event
    for phase in ['prepare','generate','score','compare']:
        if (REPORT/f'{phase}.done').exists():continue
        common.execute('l130-val500-'+phase,[sys.executable,__file__,'--phase',phase])
    event(action='controller_complete',exit_code=0)

if __name__=='__main__':
    try:main()
    except Exception as e:event(action='failed',error=repr(e));raise
