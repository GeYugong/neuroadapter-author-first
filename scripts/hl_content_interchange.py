"""Bounded fixed-L120 AA/BA/AB/BB content-source experiment; no training."""
import argparse, ast, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import evaluate_stage as ev
import lr_half_evaluate as old
ROOT=ev.ROOT;RUN=old.L
REPORT=ROOT/'reports/hl-content-interchange-l120'
WORK=RUN/'comparison/hl-content-interchange-l120'
AA=RUN/'decoded/epoch-120/val'
BB=RUN/'comparison/brain-correspondence-l120/permuted500'
KEYS=['PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV']
SOURCES={'AA':AA,'BB':BB,'BA':WORK/'BA','AB':WORK/'AB'}
METRICS={'AA':ROOT/'reports/lr-half-to120/val100','BB':ROOT/'reports/brain-correspondence-l120/permuted/val100',
         'BA':REPORT/'BA/val100','AB':REPORT/'AB/val100'}
def event(**kw):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    with (ROOT/'logs/hl-content-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def prepare():
    assert not WORK.exists()
    model=old.identity(RUN,120,63840,AA);other=old.identity(RUN,120,63840,BB)
    assert model['sha256']==other['sha256']=='bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798'
    assert model['protocol']['vae_decode_autocast']==other['protocol']['vae_decode_autocast']
    ids=old.ids('monitor100_ids.txt');full=old.ids('val_ids.txt')
    entries=old.read(ROOT/'reports/brain-correspondence-l120/brain-map.json')
    mapping={r['target_image_id']:r['brain_source_image_id'] for r in entries}
    assert len(mapping)==500 and set(mapping)==set(mapping.values())==set(full)
    assert all(i!=b for i,b in mapping.items()) and len(ids)==100
    hl=old.read(ROOT/'reports/ibbi-roi-l120-pilot/roi_mapping.json')['groups']['HL']
    assert len(set(hl))==78 and sum(i<100 for i in hl)==37 and 22 in hl and 143 in hl
    audits={r['target_image_id']:r for r in old.read(ROOT/'reports/brain-correspondence-l120/brain-input-audit.json')}
    files=[RUN/'exports/epoch-120'/x for x in ['inference.pt','metadata.json','parcel_vertices.pt']]
    files += [ROOT/x for x in ['configs/monitor100_ids.txt','configs/decode2_ids.txt','configs/atlas_val32_ids.txt',
        'reports/brain-correspondence-l120/brain-map.json','reports/brain-correspondence-l120/preflight.json',
        'reports/ibbi-roi-l120-pilot/roi_mapping.json','reports/ibbi-roi-l120-pilot/README.md',
        'reports/hl-matched-controls-l120/README.md','reports/hl-matched-controls-l120/protocol.json',
        'reports/best_so_far.json','decode_brain_adapter.py','metric_brain_adapter.py','scripts/evaluate_stage.py']]
    files += [RUN/'epochs.jsonl',AA/'protocol.json',BB/'protocol.json']
    manifest=[]
    for i in sorted(set(ids+old.ids('decode2_ids.txt'))):
        assert audits[i]['brain_source_image_id']==mapping[i]
        assert audits[i]['actual_generator_input_sha256']==audits[i]['expected_input_sha256']!=audits[i]['target_input_sha256']
        row=dict(image_id=i,donor_id=mapping[i],sources={})
        for label,folder in [('AA',AA),('BB',BB)]:
            p=folder/f'image_{i:06d}.npz';old.verify_file(p,i,AA/p.name)
            row['sources'][label]=dict(path=str(p),sha256=old.sha(p));files.append(p)
        files.append(AA/f'image_{mapping[i]:06d}.npz')
        manifest.append(row)
    for label in ['AA','BB']:
        m=old.read(METRICS[label]/'metrics.json')
        assert m['n']==100 and m['image_ids']==ids and not m['candidate_selection']
        for k in range(2):
            p=METRICS[label]/f'candidate-{k}.csv';df=pd.read_csv(p)
            assert df.image_id.tolist()==ids;files.append(p)
        files.append(METRICS[label]/'metrics.json')
    latest=json.loads((RUN/'epochs.jsonl').read_text().splitlines()[-1])
    assert latest['epoch']==140 and latest['global_optimizer_updates']==74480
    old.save(REPORT/'model_lock.json',model)
    old.save(REPORT/'reference-manifest.json',manifest)
    old.save(REPORT/'target-donor-map.json',[dict(target_image_id=i,brain_source_image_id=mapping[i]) for i in ids])
    old.save(REPORT/'hl-indices.json',dict(HL=hl,U=[i for i in range(200) if i not in hl]))
    old.save(REPORT/'input-lock.json',{str(p.relative_to(ROOT)):old.sha(p) for p in sorted(set(files))})
    old.save(REPORT/'protocol.json',dict(epoch=120,updates=63840,conditions={'AA':'HL=A,U=A (reused)','BA':'HL=B,U=A','AB':'HL=A,U=B','BB':'HL=B,U=B under target random stream (reused)'},
        n=100,candidates=2,candidate_batch=1,steps=50,cfg=4,seed=42,check_candidate_cap=8,formal_candidate_cap=400,
        training_updates=0,bootstrap_seed=2026092407,resamples=10000,primary=['D=S_AA-S_BA','R=S_AB-S_BB'],
        primary_nominal_interval=.975,auxiliary_nominal_interval=.95,
        caveat='Fixed selected model/map and reused development pool; target/donor dependence; no iid coverage, test inference, causal brain claim or resolution of infeasible spatial matching. Mixed sources need not be in training distribution.'))
    WORK.mkdir()
def command(condition,check=False):
    c=old.read(ROOT/'configs/stage1.json')
    cmd=[sys.executable,str(ROOT/'scripts/decode_hl_content.py'),'--content-condition',condition,
         '--no_selection','--split','val','--image_ids',str(ROOT/'configs'/('decode2_ids.txt' if check else 'monitor100_ids.txt')),
         '--decoded_stimuli_dir',str(WORK/(('identity-'+condition) if check else condition)),
         '--model_weights_dir',str(RUN),'--saved_epochs','120','--num_predictions','2','--noise_factor','4',
         '--precision','fp32','--batch_size','1','--denoising_steps','50','--seed','42','--num_workers','0',
         '--export_path',str(RUN/'exports/epoch-120/inference.pt')]
    for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk','condition_dim','num_decoder_queries','sub_approach']:
        cmd+=['--'+k,str(c[k])]
    return cmd
def environment(gpu):
    return {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
def execute(name,cmd,gpu=0):
    log=ROOT/f'logs/hl-content-{name}.log'
    assert not log.exists(),'Do not overwrite an existing execution log'
    event(action='start',name=name,command=cmd,gpu=gpu,log=str(log))
    with log.open('w') as f:
        result=subprocess.run(cmd,cwd=ROOT,env=environment(gpu),stdout=f,stderr=subprocess.STDOUT)
    event(action='exit',name=name,exit_code=result.returncode)
    assert result.returncode==0,(name,result.returncode,str(log))
def identity():
    rows=[]
    for label in ['AA','BB']:
        execute('identity-'+label,command(label,True))
        for i in old.ids('decode2_ids.txt'):
            with np.load(WORK/f'identity-{label}/image_{i:06d}.npz') as a,np.load(SOURCES[label]/f'image_{i:06d}.npz') as b:
                for key in ['image_id','groundtruth_image','candidate_images','seeds']:assert np.array_equal(a[key],b[key]),(label,i,key)
            rows.append(dict(condition=label,image_id=i,candidates=2,pixel_exact=True,gt_and_seed_exact=True))
    old.save(REPORT/'noninterference.json',dict(passed=True,new_check_candidates=8,checks=rows))
def generate():
    assert old.read(REPORT/'noninterference.json')['passed']
    jobs=[]
    for gpu,label in enumerate(['BA','AB']):
        log=ROOT/f'logs/hl-content-generate-{label}.log';assert not log.exists()
        stream=log.open('w');cmd=command(label)
        proc=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT)
        event(action='start',name='generate-'+label,command=cmd,pid=proc.pid,gpu=gpu,log=str(log))
        jobs.append((label,proc,stream))
    results=[]
    for label,proc,stream in jobs:
        rc=proc.wait();stream.close();event(action='exit',name='generate-'+label,exit_code=rc);results.append(rc)
    assert results==[0,0],results
    rows={}
    for label in ['BA','AB']:
        folder=SOURCES[label];old.identity(RUN,120,63840,folder)
        assert len(list(folder.glob('image_*.npz')))==100
        for i in old.ids('monitor100_ids.txt'):old.verify_file(folder/f'image_{i:06d}.npz',i,AA/f'image_{i:06d}.npz')
        rows[label]=[json.loads(x) for x in (folder/'content-audit.jsonl').read_text().splitlines()]
        assert len(rows[label])==200
    for a,b in zip(rows['BA'],rows['AB']):
        for key in ['image_id','donor_id','candidate','seed','E_A_sha256','E_B_sha256','generator_before_diffusion_sha256']:
            assert a[key]==b[key],(key,a['image_id'])
    old.save(REPORT/'content-audits.json',rows)
def score():
    for label in ['BA','AB']:ev.metrics(SOURCES[label],ROOT/'configs/monitor100_ids.txt',METRICS[label])
def direction():
    import torch
    import metric_brain_adapter as m
    # Execute only the exact author's CLIP loading/preprocessing assignments, not its metric formula.
    tree=ast.parse((ROOT/'metric_brain_adapter.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MetricEvaluator')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='compute_clip_metrics')
    assignments=[x for x in method.body if isinstance(x,ast.Assign)][:2]
    assert len(assignments)==2
    ns=dict(m.__dict__);ns['self']=m.MetricEvaluator(device='cuda')
    exec(compile(ast.Module(body=assignments,type_ignores=[]),'author_clip_setup','exec'),ns)
    model=ns['clip_model'].eval().requires_grad_(False);pre=ns['preprocessor']
    ids=old.ids('monitor100_ids.txt')
    mapping={r['target_image_id']:r['brain_source_image_id'] for r in old.read(REPORT/'target-donor-map.json')}
    def encode(t):
        with torch.no_grad():v=model.encode_image(pre(t.to('cuda'))).float()
        assert torch.isfinite(v).all()
        return torch.nn.functional.normalize(v,dim=-1).cpu()
    targets={};donors={}
    for i in ids:
        with np.load(AA/f'image_{i:06d}.npz') as z:targets[i]=encode(torch.from_numpy(z['groundtruth_image'].copy()).unsqueeze(0))
        with np.load(AA/f'image_{mapping[i]:06d}.npz') as z:donors[i]=encode(torch.from_numpy(z['groundtruth_image'].copy()).unsqueeze(0))
    rows=[]
    for label,folder in SOURCES.items():
        for i in ids:
            with np.load(folder/f'image_{i:06d}.npz') as z:
                shape=z['groundtruth_image'].shape[1:]
                for k in range(2):
                    t=torch.from_numpy(z['candidate_images'][k].copy()).permute(2,0,1).unsqueeze(0).float()/255
                    v=encode(m.resize_images(t,shape))
                    ca=float((v*targets[i]).sum());cb=float((v*donors[i]).sum())
                    rows.append(dict(condition=label,image_id=i,donor_id=mapping[i],candidate=k,cos_target=ca,cos_donor=cb,q=ca-cb))
    pd.DataFrame(rows).to_csv(REPORT/'clip-direction-candidates.csv',index=False)
    old.save(REPORT/'clip-direction-protocol.json',dict(author_setup=ast.unparse(ast.Module(body=assignments,type_ignores=[])),
        metric='Auxiliary cosine target minus cosine donor; not official two-way CLIP identification',
        clip_asset_sha256=old.sha(Path(os.environ['AUTHOR_CLIP_PATH'])),candidate_average_before_analysis=True))
def compare():
    ids=old.ids('monitor100_ids.txt');frames={};summaries={}
    for label in SOURCES:
        cs=[pd.read_csv(METRICS[label]/f'candidate-{k}.csv').set_index('image_id') for k in range(2)]
        assert all(c.index.tolist()==ids for c in cs)
        frame=(cs[0][KEYS]+cs[1][KEYS])/2;frame['S']=frame[['Alex(5)','Incep','CLIP']].mean(axis=1)
        frames[label]=frame
        met=old.read(METRICS[label]/'metrics.json')
        assert np.allclose(frame.mean(),[met['mean'][k] for k in KEYS]+[met['S']],atol=1e-10,rtol=0)
        summaries[label]=met
    q=pd.read_csv(REPORT/'clip-direction-candidates.csv')
    for label in SOURCES:
        v=q[q.condition==label].groupby('image_id')[['cos_target','cos_donor','q']].mean().loc[ids]
        frames[label]=frames[label].join(v)
    paired=pd.concat([frame.add_prefix(label+'_') for label,frame in frames.items()],axis=1)
    contrasts={'D':frames['AA']-frames['BA'],'R':frames['AB']-frames['BB']}
    contrasts['D_minus_R']=contrasts['D']-contrasts['R']
    rng=np.random.default_rng(2026092407);draws=rng.integers(0,100,size=(10000,100))
    results={}
    for name,delta in contrasts.items():
        values=delta.to_numpy();samples=np.empty((10000,values.shape[1]))
        for j in range(0,10000,100):samples[j:j+100]=values[draws[j:j+100]].mean(axis=1)
        result=[]
        for j,key in enumerate(delta.columns):
            main=key=='S' and name in ['D','R'];a=.0125 if main else .025
            ci=np.quantile(samples[:,j],[a,1-a])
            result.append(dict(metric=key,delta=float(values[:,j].mean()),low=float(ci[0]),high=float(ci[1]),
                nominal_interval=.975 if main else .95,primary=main,
                original_metric_better='lower' if key in ['Eff','SwAV'] else 'higher'))
        results[name]=result
        paired=paired.join(delta.add_prefix(name+'_'))
    paired.to_csv(REPORT/'paired-four-conditions.csv')
    old.save(REPORT/'comparison.json',dict(summaries=summaries,contrasts=results,
        direction_means={c:frames[c][['cos_target','cos_donor','q']].mean().to_dict() for c in frames},
        bootstrap_seed=2026092407,resamples=10000,pool=100,
        sampling='Average two candidates inside each target first; shared resampled target rows for all conditions; original negative pool fixed.',
        limitations=old.read(REPORT/'protocol.json')['caveat']))
def atlas():
    from PIL import Image,ImageDraw
    ids=old.ids('atlas_val32_ids.txt')
    mapping={r['target_image_id']:r['brain_source_image_id'] for r in old.read(REPORT/'target-donor-map.json')}
    out=REPORT/'atlas';out.mkdir(exist_ok=True)
    for k in range(2):
        for page in range(4):
            canvas=Image.new('RGB',(6*224,40+8*248),'white');draw=ImageDraw.Draw(canvas)
            for col,caption in enumerate(['Target GT','Donor GT','AA','BA (HL donor)','AB (HL target)','BB']):
                draw.text((col*224+4,10),caption,fill='black')
            for row,i in enumerate(ids[page*8:(page+1)*8]):
                with np.load(AA/f'image_{i:06d}.npz') as z:a=(z['groundtruth_image'].transpose(1,2,0)*255).round().astype('uint8')
                with np.load(AA/f'image_{mapping[i]:06d}.npz') as z:b=(z['groundtruth_image'].transpose(1,2,0)*255).round().astype('uint8')
                images=[a,b]
                for label in ['AA','BA','AB','BB']:
                    with np.load(SOURCES[label]/f'image_{i:06d}.npz') as z:images.append(z['candidate_images'][k].copy())
                for col,img in enumerate(images):canvas.paste(Image.fromarray(img).resize((224,224)),(col*224,40+row*248))
                draw.text((4,40+row*248+226),f'target {i}, donor {mapping[i]}, candidate {k}',fill='black')
            canvas.save(out/f'candidate-{k}-page-{page+1}.jpg',quality=92)
def finish():
    checks={p:old.sha(ROOT/p)==h for p,h in old.read(REPORT/'input-lock.json').items()}
    assert all(checks.values())
    counts={p.name:len(list(p.glob('image_*.npz')))*2 for p in WORK.iterdir() if p.is_dir()}
    assert counts=={'identity-AA':4,'identity-BB':4,'BA':200,'AB':200},counts
    latest=json.loads((RUN/'epochs.jsonl').read_text().splitlines()[-1])
    assert (latest['epoch'],latest['global_optimizer_updates'])==(140,74480)
    old.save(REPORT/'completion.json',dict(key_input_hashes_unchanged=checks,candidate_counts=counts,
        new_candidates=408,new_training_updates=0,latest_progress=latest,
        compute_complete=True,AI_review_and_publication_pending=True))
    (REPORT/'EVALUATION_COMPLETE').write_text('Computed; actual fixed atlas review and Chinese report/publication pending.\n')
def main():
    phases=['prepare','identity','generate','score','direction','compare','atlas','finish']
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=phases);args=p.parse_args()
    if args.phase:
        globals()[args.phase]();(REPORT/(args.phase+'.done')).write_text('exit0\n');return
    lock=(ROOT/'runtime/hl-content.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for phase in phases:
        if (REPORT/(phase+'.done')).exists():continue
        execute(phase,[sys.executable,__file__,'--phase',phase])
    event(action='controller_complete',exit_code=0)
if __name__=='__main__':
    try:main()
    except Exception as e:event(action='failed',error=repr(e));raise
