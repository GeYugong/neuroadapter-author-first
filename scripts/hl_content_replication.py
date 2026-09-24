"""Fixed new-target/three-donor replication; author generation and evaluation are reused."""
import argparse, concurrent.futures, fcntl, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import evaluate_stage as ev
import lr_half_evaluate as old
import hl_content_interchange as pilot
ROOT=ev.ROOT;RUN=old.L;AA=pilot.AA
REPORT=ROOT/'reports/hl-content-replication-l120'
WORK=RUN/'comparison/hl-content-replication-l120'
KEYS=pilot.KEYS
def event(**kw):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    with (ROOT/'logs/hl-replication-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def targets():return np.loadtxt(REPORT/'target_ids.txt',dtype=int).reshape(-1).tolist()
def mapping(m):return {x['target_image_id']:x['brain_source_image_id'] for x in old.read(REPORT/f'donor-{m}/map.json')}
def sources(m):return {'AA':AA,**{x:WORK/f'donor-{m}'/x for x in ['BA','AB','BB']}}
def config(mode,m=None):
    mapfile=ROOT/'reports/brain-correspondence-l120/brain-map.json' if mode=='check' else REPORT/f'donor-{m}/map.json'
    return dict(study='L120-HL-content-replication',mode=mode,
        allowed_conditions=['AA','BB'] if mode=='check' else ['BA','AB','BB'],
        target_ids=str(ROOT/'configs/decode2_ids.txt' if mode=='check' else REPORT/'target_ids.txt'),
        mapping_path=str(mapfile),mapping_sha256=old.sha(mapfile),
        output_root=str(WORK/'checks' if mode=='check' else WORK/f'donor-{m}'),
        roi_mapping_sha256=old.sha(ROOT/'reports/ibbi-roi-l120-pilot/roi_mapping.json'),
        model_sha256='bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798')
def prepare():
    assert not WORK.exists() and not (REPORT/'target_ids.txt').exists()
    full=old.ids('val_ids.txt');prior=old.ids('monitor100_ids.txt')
    assert len(full)==500 and len(set(full))==500 and len(prior)==100
    oldmap=old.read(ROOT/'reports/hl-content-interchange-l120/target-donor-map.json')
    assert [x['target_image_id'] for x in oldmap]==prior
    remaining=np.array(sorted(set(full)-set(prior)),dtype=np.int64);assert len(remaining)==400
    perm=np.random.default_rng(2026092408).permutation(remaining)
    t=perm[:100].tolist();blocks=[perm[100:200].tolist(),perm[200:300].tolist(),perm[300:400].tolist()]
    assert len(set(t+sum(blocks,[])))==400
    np.savetxt(REPORT/'target_ids.txt',t,fmt='%d')
    np.savetxt(REPORT/'atlas32_ids.txt',t[:32],fmt='%d')
    allrows=[]
    for m,donors in enumerate(blocks,1):
        folder=REPORT/f'donor-{m}';folder.mkdir()
        rows=[dict(target_image_id=i,brain_source_image_id=b) for i,b in zip(t,donors)]
        old.save(folder/'map.json',rows)
        allrows += [dict(batch=m,**x) for x in rows]
        old.save(folder/'study-config.json',config('formal',m))
    old.save(REPORT/'check-config.json',config('check'))
    pd.DataFrame(allrows).to_csv(REPORT/'all-target-donor-pairs.csv',index=False)
    identity=old.identity(RUN,120,63840,AA)
    assert identity['sha256']==config('check')['model_sha256']
    hl=old.read(ROOT/'reports/ibbi-roi-l120-pilot/roi_mapping.json')['groups']['HL']
    assert len(set(hl))==78 and sum(i<100 for i in hl)==37 and 22 in hl and 143 in hl
    old.save(REPORT/'model-lock.json',identity);old.save(REPORT/'hl-indices.json',dict(HL=hl,U=[i for i in range(200) if i not in hl]))
    files=[RUN/'exports/epoch-120'/x for x in ['inference.pt','metadata.json','parcel_vertices.pt']]
    files += [ROOT/x for x in ['configs/val_ids.txt','configs/monitor100_ids.txt','configs/decode2_ids.txt',
        'reports/best_so_far.json','reports/ibbi-roi-l120-pilot/roi_mapping.json','reports/ibbi-roi-l120-pilot/README.md',
        'reports/hl-matched-controls-l120/README.md','reports/hl-content-interchange-l120/README.md',
        'reports/brain-correspondence-l120/README.md','reports/hl-content-interchange-l120/target-donor-map.json',
        'decode_brain_adapter.py','metric_brain_adapter.py','scripts/evaluate_stage.py']]
    files+=[RUN/'epochs.jsonl',AA/'protocol.json']
    manifest=[]
    for i in sorted(set(perm.tolist()+old.ids('decode2_ids.txt'))):
        p=AA/f'image_{i:06d}.npz';old.verify_file(p,i);files.append(p)
        if i in t:manifest.append(dict(image_id=i,path=str(p),sha256=old.sha(p)))
    for i in old.ids('decode2_ids.txt'):files.append(pilot.BB/f'image_{i:06d}.npz')
    files.append(ROOT/'reports/brain-correspondence-l120/brain-map.json')
    files+=[REPORT/'target_ids.txt',REPORT/'atlas32_ids.txt',REPORT/'all-target-donor-pairs.csv',REPORT/'check-config.json']
    for m in range(1,4):files += [REPORT/f'donor-{m}/map.json',REPORT/f'donor-{m}/study-config.json']
    old.save(REPORT/'AA-reuse-manifest.json',sorted(manifest,key=lambda x:t.index(x['image_id'])))
    old.save(REPORT/'input-lock.json',{str(p.relative_to(ROOT)):old.sha(p) for p in sorted(set(files))})
    latest=json.loads((RUN/'epochs.jsonl').read_text().splitlines()[-1]);assert (latest['epoch'],latest['global_optimizer_updates'])==(140,74480)
    old.save(REPORT/'protocol.json',dict(partition_seed=2026092408,bootstrap_seed=2026092409,
        sorted_remaining=remaining.tolist(),permutation=perm.tolist(),new_target_count=100,donor_count=300,
        mutually_disjoint_ids=True,formal_candidates=1800,check_candidate_cap=8,training_updates=0,
        candidate_batch=1,steps=50,cfg=4,candidates_per_target=2,primary=['D_bar=mean_targets(mean_donors(AA-BA))','R_bar=mean_targets(mean_donors(AB-BB))'],
        bootstrap='10000 shared resamples of 100 fixed target+three-donor blocks after equal candidate and donor averaging',
        primary_nominal_interval=.975,auxiliary_nominal_interval=.95,
        limits='Repeatedly used development pool, fixed selected L120 and fixed donor partition; not independent test, population causality, donor-general coverage or solution to spatial matching. Mixed-token joint distributions may be off-distribution.'))
    WORK.mkdir()
def command(label,m=None,check=False):
    cmd=pilot.command(label,check)
    def setarg(k,v):cmd[cmd.index(k)+1]=str(v)
    setarg('--image_ids',ROOT/'configs/decode2_ids.txt' if check else REPORT/'target_ids.txt')
    setarg('--decoded_stimuli_dir',WORK/'checks'/label if check else WORK/f'donor-{m}'/label)
    cmd+=['--study-config',str(REPORT/'check-config.json' if check else REPORT/f'donor-{m}/study-config.json')]
    return cmd
def execute(name,cmd,gpu=0):
    log=ROOT/f'logs/hl-replication-{name}.log';assert not log.exists(),str(log)
    event(action='start',name=name,command=cmd,gpu=gpu,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    with log.open('w') as f:
        result=subprocess.run(cmd,cwd=ROOT,env=pilot.environment(gpu),stdout=f,stderr=subprocess.STDOUT)
    event(action='exit',name=name,exit_code=result.returncode)
    assert result.returncode==0,(name,result.returncode,str(log))
def identity():
    records=[]
    for label in ['AA','BB']:
        execute('identity-'+label,command(label,check=True))
        for i in old.ids('decode2_ids.txt'):
            with np.load(WORK/f'checks/{label}/image_{i:06d}.npz') as a,np.load((AA if label=='AA' else pilot.BB)/f'image_{i:06d}.npz') as b:
                for key in ['image_id','groundtruth_image','candidate_images','seeds']:assert np.array_equal(a[key],b[key])
            records.append(dict(condition=label,image_id=i,both_candidates_pixel_exact=True,gt_and_seeds_exact=True))
    old.save(REPORT/'noninterference.json',dict(new_candidates=8,checks=records,passed=True))
def generate():
    assert old.read(REPORT/'noninterference.json')['passed']
    jobs=[(m,c) for m in range(1,4) for c in ['BA','AB','BB']]
    def worker(gpu):
        for m,label in jobs[gpu::2]:
            done=REPORT/f'donor-{m}/{label}-generated.done'
            if done.exists():continue
            execute(f'donor-{m}-{label}',command(label,m),gpu)
            done.write_text('exit0\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures=[ex.submit(worker,g) for g in range(2)]
        for f in futures:f.result()
    audits={};manifest=[];t=targets()
    for m in range(1,4):
        audits[str(m)]={}
        for label in ['BA','AB','BB']:
            folder=sources(m)[label];old.identity(RUN,120,63840,folder)
            assert len(list(folder.glob('image_*.npz')))==100
            rows=[json.loads(x) for x in (folder/'content-audit.jsonl').read_text().splitlines()]
            assert [(x['image_id'],x['candidate']) for x in rows]==[(i,k) for i in t for k in range(2)]
            assert all(x['donor_id']==mapping(m)[x['image_id']] for x in rows)
            audits[str(m)][label]=rows
            for i in t:
                p=folder/f'image_{i:06d}.npz';old.verify_file(p,i,AA/p.name)
                manifest.append(dict(batch=m,condition=label,image_id=i,donor_id=mapping(m)[i],candidates=2,path=str(p),sha256=old.sha(p)))
        for rows in zip(*[audits[str(m)][x] for x in ['BA','AB','BB']]):
            for key in ['image_id','donor_id','candidate','seed','E_A_sha256','E_B_sha256','generator_before_diffusion_sha256']:
                assert len({r[key] for r in rows})==1,key
    for rows in zip(*[audits[str(m)]['BA'] for m in range(1,4)]):
        for key in ['image_id','candidate','seed','E_A_sha256','generator_before_diffusion_sha256']:
            assert len({r[key] for r in rows})==1,key
    old.save(REPORT/'content-audits.json',audits);old.save(REPORT/'generation-manifest.json',manifest)
def score():
    ev.metrics(AA,REPORT/'target_ids.txt',REPORT/'AA/val100')
    for m in range(1,4):
        for label in ['BA','AB','BB']:ev.metrics(sources(m)[label],REPORT/'target_ids.txt',REPORT/f'donor-{m}/{label}/val100')
def direction():
    for m in range(1,4):pilot.direction_for(targets(),mapping(m),sources(m),REPORT/f'donor-{m}',AA)
def compare():
    t=targets();cols=KEYS+['S'];rng=np.random.default_rng(2026092409);draws=rng.integers(0,100,size=(10000,100))
    def load(folder):
        cs=[pd.read_csv(folder/f'candidate-{k}.csv').set_index('image_id') for k in range(2)]
        assert all(x.index.tolist()==t for x in cs)
        for x in cs:x['S']=x[['Alex(5)','Incep','CLIP']].mean(axis=1)
        mean=(cs[0][cols]+cs[1][cols])/2
        met=old.read(folder/'metrics.json')
        assert np.allclose(mean.mean(),[met['mean'][k] for k in KEYS]+[met['S']],rtol=0,atol=1e-10)
        mean.to_csv(folder/'per-target-mean.csv')
        return mean,cs,met
    def intervals(df,primary=False):
        values=df.to_numpy();sample=np.empty((10000,len(df.columns)))
        for j in range(0,10000,100):sample[j:j+100]=values[draws[j:j+100]].mean(axis=1)
        result=[]
        for j,k in enumerate(df.columns):
            a=.0125 if primary and k=='S' else .025;ci=np.quantile(sample[:,j],[a,1-a])
            result.append(dict(metric=k,delta=float(values[:,j].mean()),low=float(ci[0]),high=float(ci[1]),
                nominal_interval=1-2*a,primary=bool(primary and k=='S')))
        return result
    aa,aac,aa_summary=load(REPORT/'AA/val100');by_batch={};D=[];R=[];candidate_rows=[];frame_rows=[]
    for m in range(1,4):
        frames={'AA':aa.copy()};candidates={'AA':aac};summaries={'AA':aa_summary}
        for label in ['BA','AB','BB']:
            frames[label],candidates[label],summaries[label]=load(REPORT/f'donor-{m}/{label}/val100')
        q=pd.read_csv(REPORT/f'donor-{m}/clip-direction-candidates.csv')
        for label in frames:
            aux=q[q.condition==label].groupby('image_id')[['cos_target','cos_donor','q']].mean().loc[t]
            frames[label]=frames[label].join(aux)
        d=frames['AA']-frames['BA'];r=frames['AB']-frames['BB'];D.append(d);R.append(r)
        joined=pd.concat([v.add_prefix(k+'_') for k,v in frames.items()]+[d.add_prefix('D_'),r.add_prefix('R_')],axis=1)
        joined.insert(0,'donor_id',[mapping(m)[i] for i in t]);joined.to_csv(REPORT/f'donor-{m}/paired-results.csv')
        for k in range(2):
            for name,x,y in [('D','AA','BA'),('R','AB','BB')]:
                delta=candidates[x][k][cols]-candidates[y][k][cols]
                candidate_rows.append(dict(batch=m,candidate=k,contrast=name,**delta.mean().to_dict()))
        by_batch[str(m)]=dict(summaries=summaries,D=intervals(d),R=intervals(r),
            direction_means={k:v[['cos_target','cos_donor','q']].mean().to_dict() for k,v in frames.items()})
    dbar=sum(D)/3;rbar=sum(R)/3
    joined=pd.concat([dbar.add_prefix('D_mean_'),rbar.add_prefix('R_mean_')],axis=1)
    for name,frames in [('D',D),('R',R)]:
        for col in frames[0].columns:
            arr=np.stack([f[col].to_numpy() for f in frames],axis=1)
            joined[name+'_'+col+'_donor_min']=arr.min(axis=1);joined[name+'_'+col+'_donor_max']=arr.max(axis=1)
            for m in range(3):joined[f'{name}_{col}_donor_{m+1}']=arr[:,m]
    joined.to_csv(REPORT/'target-block-results.csv')
    pd.DataFrame(candidate_rows).to_csv(REPORT/'candidate-contrast-trends.csv',index=False)
    old.save(REPORT/'comparison.json',dict(by_batch=by_batch,aggregate=dict(D=intervals(dbar,True),R=intervals(rbar,True),
        D_minus_R=intervals(dbar-rbar)),bootstrap_seed=2026092409,resamples=10000,
        unit='100 fixed target+three-donor blocks; first equal average of two candidates and three donors; common bootstrap draws',
        limits=old.read(REPORT/'protocol.json')['limits']))
def atlas():
    t=targets();fixed=np.loadtxt(REPORT/'atlas32_ids.txt',dtype=int).tolist();assert fixed==t[:32]
    lines=['# 新100目标全部候选浏览索引','','前4页为预先固定32图；第5至13页展示其余目标。三个批次、两个候选全部保留。','']
    for m in range(1,4):
        pilot.atlas_for(t,mapping(m),sources(m),REPORT/f'donor-{m}/atlas',AA)
        for k in range(2):
            lines += [f'## 供体批次{m}，候选{k}','']
            for page in range(1,14):
                ids=t[(page-1)*8:page*8]
                lines.append(f'- [第{page}页：'+', '.join(map(str,ids))+f'](donor-{m}/atlas/candidate-{k}-page-{page}.jpg)')
    (REPORT/'ALL_TARGETS.md').write_text('\n'.join(lines)+'\n')
def finish():
    locks=old.read(REPORT/'input-lock.json');checks={p:old.sha(ROOT/p)==h for p,h in locks.items()};assert all(checks.values())
    count=sum(len(list((WORK/f'donor-{m}'/c).glob('image_*.npz')))*2 for m in range(1,4) for c in ['BA','AB','BB'])
    check_count=sum(len(list((WORK/'checks'/c).glob('image_*.npz')))*2 for c in ['AA','BB'])
    assert count==1800 and check_count==8
    old.save(REPORT/'completion.json',dict(formal_candidates=count,check_candidates=check_count,new_training_updates=0,
        input_hashes_unchanged=checks,compute_complete=True,AI_review_and_publication_pending=True))
    (REPORT/'EVALUATION_COMPLETE').write_text('Computation complete; actual 24-page AI review and report/publication pending.\n')
def main():
    phases=['prepare','identity','generate','score','direction','compare','atlas','finish']
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=phases);args=p.parse_args()
    if args.phase:
        globals()[args.phase]();(REPORT/(args.phase+'.done')).write_text('exit0\n');return
    lock=(ROOT/'runtime/hl-replication.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in phases:
        if (REPORT/(name+'.done')).exists():continue
        execute(name,[sys.executable,__file__,'--phase',name])
    event(action='controller_complete',exit_code=0)
if __name__=='__main__':
    try:main()
    except Exception as e:event(action='failed',error=repr(e));raise
