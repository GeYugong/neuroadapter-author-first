"""Bounded L120 IBBI/ROI orchestration; author decode/metric functions remain unchanged."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import evaluate_stage as ev
import lr_half_evaluate as old
from ibbi_roi_prepare import ROOT,RUN,REPORT,WORK
BASE=ROOT/'reports/lr-half-to120'
KEYS=['PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV']
CONDITIONS=['LL','HL','LL-R1','HL-R1','LL-R2','HL-R2','LL-R3','HL-R3']
def event(**kw):
    row={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw}
    with (ROOT/'logs/ibbi-roi-l120-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def launch(name,cmd,gpu='0'):
    log=ROOT/'logs'/f'ibbi-roi-l120-{name}.log'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
    assert not log.exists(),'Existing log: require evidence-based resume, not an implicit rerun'
    handle=log.open('w')
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    event(action='start',name=name,pid=proc.pid,gpu=gpu,command=cmd,log=str(log.relative_to(ROOT)))
    return name,proc,handle
def finish(job):
    name,proc,handle=job;code=proc.wait();handle.close();event(action='exit',name=name,pid=proc.pid,exit_code=code)
    return code
def run_jobs(jobs):
    codes=[finish(job) for job in jobs]
    assert all(c==0 for c in codes),codes
def decode_cmd(condition,ids_path,out,record=False):
    c=old.read(ROOT/'configs/stage1.json')
    cmd=[sys.executable,str(ROOT/'scripts/decode_ibbi_roi.py'),'--pilot-condition',condition]
    if record:cmd+=['--record-attention']
    cmd+=['--no_selection','--split','val','--image_ids',str(ids_path),'--decoded_stimuli_dir',str(out),
          '--model_weights_dir',str(RUN),'--saved_epochs','120','--num_predictions','2','--noise_factor','4',
          '--precision','fp32','--batch_size','1','--denoising_steps','50','--seed','42','--num_workers','0',
          '--export_path',str(RUN/'exports/epoch-120/inference.pt')]
    for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk','condition_dim','num_decoder_queries','sub_approach']:
        cmd+=['--'+k,str(c[k])]
    return cmd
def verify_output(folder,ids,baseline_exact=False):
    assert len(list(folder.glob('image_*.npz')))==len(ids)
    for i in ids:
        old.verify_file(folder/f'image_{i:06d}.npz',i,RUN/f'decoded/epoch-120/val/image_{i:06d}.npz')
        if baseline_exact:
            with np.load(folder/f'image_{i:06d}.npz') as a,np.load(RUN/f'decoded/epoch-120/val/image_{i:06d}.npz') as b:
                assert all(np.array_equal(a[k],b[k]) for k in ['image_id','groundtruth_image','candidate_images','seeds'])
    audit=[json.loads(s) for s in (folder/'input-audit.jsonl').read_text().splitlines()]
    assert [(r['image_id'],r['candidate']) for r in audit]==[(i,k) for i in ids for k in (0,1)]
    assert all(r['unselected_rows_exact'] and r['selected_rows_zero'] and r['conditional_checks']==50 for r in audit)
    return audit
def identity():
    lock=old.read(REPORT/'model_lock.json')
    for p,h in lock['files'].items():assert old.sha(ROOT/p)==h
    run_jobs([launch('identity',decode_cmd('BASE',REPORT/'check2_ids.txt',WORK/'identity2',True))])
    rows=verify_output(WORK/'identity2',old.ids('decode2_ids.txt'),True)
    old.save(REPORT/'noninterference.json',{'passed':True,'attention_and_explicit_empty_mask_checked_together':True,
       'original_decoder_same_forward':True,'images':2,'candidates':4,'rows':rows,
       'weight_unchanged':{p:old.sha(ROOT/p)==h for p,h in lock['files'].items()}})
def ibbi():
    assert old.read(REPORT/'noninterference.json')['passed']
    remaining=np.loadtxt(REPORT/'ibbi_remaining_ids.txt',dtype=int).tolist()
    jobs=[];folders=[(WORK/'identity2',old.ids('decode2_ids.txt'))]
    for rank,part in enumerate(np.array_split(remaining,2)):
        ids=[int(x) for x in part];path=REPORT/f'ibbi-shard{rank}_ids.txt'
        path.write_text(''.join(str(i)+'\n' for i in ids));out=WORK/f'ibbi-shard{rank}'
        folders.append((out,ids));jobs.append(launch(f'ibbi-shard{rank}',decode_cmd('BASE',path,out,True),str(rank)))
    run_jobs(jobs)
    full=WORK/'ibbi32';full.mkdir();(full/'attention').mkdir()
    rows=[]
    for folder,ids in folders:
        rows.extend(verify_output(folder,ids,True))
        for i in ids:
            (full/f'image_{i:06d}.npz').symlink_to(folder/f'image_{i:06d}.npz')
            for k in (0,1):
                for ext in ('npz','json'):
                    source=folder/f'attention/image_{i:06d}-candidate-{k}.{ext}'
                    assert source.exists();(full/'attention'/source.name).symlink_to(source)
    assert len(list((full/'attention').glob('*.npz')))==64
    old.save(REPORT/'ibbi-validation.json',{'trajectories':64,'all_candidates_baseline_pixel_exact':True,
        'all_conditional_rows_verified':True,'trace_directory':str(full/'attention'),'rows':rows})
def figures():
    from ibbi_roi_figures import attention_figures
    attention_figures()
def masks():
    assert (REPORT/'figures.done').exists() and old.read(REPORT/'noninterference.json')['passed']
    available=old.read(REPORT/'roi_mapping.json')['mask_sets']
    for start in range(0,len(CONDITIONS),2):
        jobs=[]
        for rank,name in enumerate(CONDITIONS[start:start+2]):
            if name not in available:continue
            jobs.append(launch('generate-'+name,decode_cmd(name,REPORT/'pilot100_ids.txt',WORK/name),str(rank)))
        run_jobs(jobs)
    for name in CONDITIONS:
        if name in available:
            audit=verify_output(WORK/name,old.ids('monitor100_ids.txt'))
            old.save(REPORT/name/'input-audit.json',audit)
def score_one(name):
    assert name in CONDITIONS
    ev.metrics(WORK/name,REPORT/'pilot100_ids.txt',REPORT/name/'val100')
    ev.atlas(WORK/name,REPORT/'ibbi32_ids.txt',REPORT/name/'val-atlas')
def score():
    available=old.read(REPORT/'roi_mapping.json')['mask_sets']
    for start in range(0,len(CONDITIONS),2):
        jobs=[]
        for rank,name in enumerate(CONDITIONS[start:start+2]):
            if name in available:jobs.append(launch('score-'+name,[sys.executable,__file__,'--score-condition',name],str(rank)))
        run_jobs(jobs)
    from ibbi_roi_figures import intervention_figures
    intervention_figures()
def compare():
    ids=old.ids('monitor100_ids.txt');frames={};summaries={}
    names=['BASE']+[n for n in CONDITIONS if n in old.read(REPORT/'roi_mapping.json')['mask_sets']]
    for name in names:
        path=BASE/'val100' if name=='BASE' else REPORT/name/'val100'
        cs=[pd.read_csv(path/f'candidate-{k}.csv').set_index('image_id') for k in range(2)]
        assert all(len(c)==100 and c.index.is_unique and set(c.index)==set(ids) for c in cs)
        frame=(cs[0].loc[ids,KEYS]+cs[1].loc[ids,KEYS])/2
        frame['S']=frame[['Alex(5)','Incep','CLIP']].mean(axis=1)
        frames[name]=frame;summaries[name]=old.read(path/'metrics.json')
        assert np.allclose(frame.mean(),[summaries[name]['mean'][k] for k in KEYS]+[summaries[name]['S']],atol=1e-10,rtol=0)
        frame.to_csv(REPORT/f'per-image-{name}.csv')
    # Same image-level paired percentile implementation as complete_l130_val500.py.
    draws=np.random.default_rng(2026092404).integers(0,100,size=(10000,100))
    contrasts={}
    differences={f'D_{n}':frames['BASE']-frames[n] for n in names if n!='BASE'}
    for roi in ['LL','HL']:
        if all(roi+f'-R{k}' in names for k in (1,2,3)):
            random_mean=sum(differences[f'D_{roi}-R{k}'] for k in (1,2,3))/3
            differences[f'D_{roi}-Rmean']=random_mean
            differences[f'E_{roi}']=differences[f'D_{roi}']-random_mean
    for name,delta in differences.items():
        delta.to_csv(REPORT/f'paired-{name}.csv')
        values=delta.to_numpy();assert np.isfinite(values).all()
        samples=np.empty((10000,9))
        for start in range(0,10000,100):samples[start:start+100]=values[draws[start:start+100]].mean(axis=1)
        rows=[]
        for j,k in enumerate(KEYS+['S']):
            level=.975 if name in ['E_LL','E_HL'] and k=='S' else .95
            lo,hi=np.quantile(samples[:,j],[(1-level)/2,1-(1-level)/2])
            rows.append({'metric':k,'delta':float(values[:,j].mean()),'low':float(lo),'high':float(hi),
                'interval_level':level,'original_metric_better':'lower' if k in ['Eff','SwAV'] else 'higher'})
        contrasts[name]=rows
    old.save(REPORT/'comparison.json',{'summaries':summaries,'contrasts':contrasts,'n_images':100,'resamples':10000,
       'seed':2026092404,'D':'BASE minus mask','E':'D_ROI minus mean of its three random-set D values',
       'scope':'Descriptive fixed model, masks and reused internal development set; no independent-test, historical-selection correction, brain causality or randomized population claim.'})
    unchanged={}
    for p,h in old.read(REPORT/'protected-before.json').items():
        unchanged[p]=old.sha(ROOT/p)==h
        assert unchanged[p],p
    progress=old.read(RUN/'checkpoint-140/progress.json')
    old.save(REPORT/'completion.json',{'training_updates_added':0,'training_latest_progress':progress,
       'ibbi_trajectories':64,'check_trajectories_reused':4,'intervention_images':100*(len(names)-1),
       'intervention_candidates':200*(len(names)-1),'protected_files':len(unchanged),'all_protected_hashes_unchanged':True})
    (REPORT/'EVALUATION_COMPLETE').write_text('All fixed computations complete; final AI atlas review and Chinese report pending.\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['identity','ibbi','figures','masks','score','compare'])
    p.add_argument('--score-condition',choices=CONDITIONS);a=p.parse_args()
    if a.score_condition:score_one(a.score_condition);return
    if a.phase:
        globals()[a.phase]();(REPORT/f'{a.phase}.done').write_text('exit0\n');return
    lock=(ROOT/'runtime/ibbi-roi-l120.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (REPORT/'PLAN.md').exists()
    for phase in ['identity','ibbi','figures','masks','score','compare']:
        if (REPORT/f'{phase}.done').exists():continue
        run_jobs([launch('phase-'+phase,[sys.executable,__file__,'--phase',phase])])
    event(action='controller_complete',exit_code=0)
if __name__=='__main__':
    try:main()
    except Exception as e:event(action='failed',error=repr(e));raise

