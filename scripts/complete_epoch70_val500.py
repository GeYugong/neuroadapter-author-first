"""Evaluation-only completion of epoch70; original decoder/metrics are reused."""
import fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import evaluate_stage as evaluation
ROOT=evaluation.ROOT
RUN=ROOT/'runs/subject01-author-fp32-seed42'
WORK=RUN/'decoded/epoch-70-val500-supplement'
REPORT=ROOT/'reports/epoch-70/val500'
COMPARE=ROOT/'reports/epoch-70-vs-80-val500'
SEED=20260922

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def event(action,**kw):
    record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),action=action,**kw)
    print(json.dumps(record),flush=True)
def main():
    lock=(ROOT/'runtime/epoch70-val500.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not WORK.exists() and not REPORT.exists() and not COMPARE.exists(), 'Existing supplement: inspect; do not restart or overwrite'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    export=RUN/'exports/epoch-70/inference.pt';meta=json.loads(export.with_name('metadata.json').read_text())
    assert meta['epoch']==70 and meta['global_optimizer_updates']==37240 and meta['sha256']==sha(export)
    protocol=json.loads((RUN/'decoded/epoch-70/val/protocol.json').read_text())
    assert Path(protocol['export_path'])==export
    for k,v in dict(split='val',candidate_batch=1,num_predictions=2,denoising_steps=50,cfg=4,seed=42,selection=False,gt_initialization=False,weight_dtype='torch.float32').items():assert protocol[k]==v
    historical=json.loads((ROOT/'reports/epoch-70/delivery-checks.json').read_text())
    assert historical['hashes']['runs/subject01-author-fp32-seed42/exports/epoch-70']['inference.pt']==sha(export)
    full=np.loadtxt(ROOT/'configs/val_ids.txt',dtype=int).tolist();old=np.loadtxt(ROOT/'configs/monitor100_ids.txt',dtype=int).tolist()
    assert len(full)==500 and len(set(full))==500 and len(old)==100 and set(old)<set(full)
    old_hash={}
    for i in old:
        f=RUN/f'decoded/epoch-70/val/image_{i:06d}.npz'
        with np.load(f) as z:
            assert int(z['image_id'])==i and len(z['candidate_images'])==2
            assert list(z['seeds'])==[int.from_bytes(hashlib.sha256(f'42:{i}:{k}'.encode()).digest()[:8],'little')%(2**63-1) for k in range(2)]
        old_hash[str(i)]=sha(f)
    missing=[i for i in full if i not in set(old)];assert len(missing)==400
    protected=[ROOT/'reports/best_so_far.json',RUN/'best_so_far/inference.pt',RUN/'best_so_far/metadata.json']
    protected += list(RUN.glob('checkpoint-*/progress.json')) + list(RUN.glob('exports/epoch-*/inference.pt'))
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    WORK.mkdir();REPORT.mkdir(parents=True);COMPARE.mkdir()
    np.savetxt(WORK/'missing400_ids.txt',missing,fmt='%d');np.savetxt(REPORT/'image_ids.txt',full,fmt='%d')
    save(COMPARE/'provenance.json',dict(epoch=70,updates=37240,export_path=str(export),export_sha256=sha(export),existing100_protocol=protocol,existing100_sha256=old_hash,protected_sha256=before,new_images=400,new_candidates=800,bootstrap_seed=SEED,bootstrap_resamples=10000))
    c=json.loads((ROOT/'configs/stage1.json').read_text());jobs=[]
    for gpu,ids in enumerate([missing[:200],missing[200:]]):
        ids_path=WORK/f'missing200-gpu{gpu}.txt';np.savetxt(ids_path,ids,fmt='%d')
        cmd=[sys.executable,str(ROOT/'decode_brain_adapter.py'),'--no_selection','--split','val','--image_ids',str(ids_path),'--decoded_stimuli_dir',str(WORK/f'gpu{gpu}'),'--model_weights_dir',str(RUN),'--saved_epochs','70','--num_predictions','2','--noise_factor','4','--precision','fp32','--batch_size','1','--denoising_steps','50','--seed','42','--num_workers','0','--export_path',str(export)]
        for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk','condition_dim','num_decoder_queries','sub_approach']:cmd+=['--'+k,str(c[k])]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'}
        log=(ROOT/f'logs/epoch70-val500-gpu{gpu}.log').open('w')
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((gpu,proc,log));event('decode_launch',gpu=gpu,pid=proc.pid,command=cmd)
    codes=[]
    for gpu,proc,log in jobs:
        code=proc.wait();log.close();codes.append(code);event('decode_exit',gpu=gpu,exit_code=code)
    assert codes==[0,0],codes
    union=WORK/'full500';union.mkdir();manifest=[]
    for i in full:
        source=RUN/f'decoded/epoch-70/val/image_{i:06d}.npz' if i in set(old) else WORK/('gpu0' if i in missing[:200] else 'gpu1')/f'image_{i:06d}.npz'
        with np.load(source) as z,np.load(RUN/f'decoded/epoch-80/val/image_{i:06d}.npz') as ref:
            assert int(z['image_id'])==i and len(z['candidate_images'])==2
            assert np.array_equal(z['groundtruth_image'],ref['groundtruth_image']) and np.array_equal(z['seeds'],ref['seeds'])
        if str(i) in old_hash:assert sha(source)==old_hash[str(i)]
        (union/source.name).symlink_to(source)
        manifest.append(dict(image_id=i,source=str(source),sha256=sha(source),reused=i in set(old)))
    save(COMPARE/'full500-manifest.json',manifest)
    event('scoring_start',n=500)
    evaluation.metrics(union,REPORT/'image_ids.txt',REPORT)
    event('scoring_complete')
    keys=['PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV']
    frames={}
    for ep in [70,80]:
        fs=[pd.read_csv(ROOT/f'reports/epoch-{ep}/val500/candidate-{k}.csv').set_index('image_id').loc[full,keys] for k in [0,1]]
        frames[ep]=(fs[0]+fs[1])/2
        frames[ep]['S']=frames[ep][['Alex(5)','Incep','CLIP']].mean(axis=1)
        summary=json.loads((ROOT/f'reports/epoch-{ep}/val500/metrics.json').read_text())
        assert np.allclose(frames[ep].mean().values,[summary['mean'][k] for k in keys]+[summary['S']],rtol=0,atol=1e-10)
    delta=frames[70]-frames[80]
    paired=pd.concat([frames[70].add_prefix('epoch70_'),frames[80].add_prefix('epoch80_'),delta.add_prefix('delta70_minus80_')],axis=1)
    paired.to_csv(COMPARE/'paired-image-differences.csv')
    rng=np.random.default_rng(SEED);values=delta.to_numpy();samples=np.empty((10000,9))
    for start in range(0,10000,100):
        idx=rng.integers(0,500,size=(100,500));samples[start:start+100]=values[idx].mean(axis=1)
    ci=np.quantile(samples,[.025,.975],axis=0);results=[]
    for j,k in enumerate(keys+['S']):
        results.append(dict(metric=k,epoch70=float(frames[70][k].mean()),epoch80=float(frames[80][k].mean()),delta70_minus80=float(values[:,j].mean()),ci95_low=float(ci[0,j]),ci95_high=float(ci[1,j]),better_direction='lower' if k in ['Eff','SwAV'] else 'higher'))
    pd.DataFrame(results).to_csv(COMPARE/'comparison.csv',index=False)
    save(COMPARE/'bootstrap.json',dict(seed=SEED,resamples=10000,n_images=500,method='paired image-level percentile bootstrap; two candidates averaged within image before resampling; same image indices for epochs and metrics',identification_pool='Per-image identification scores retain the original full500 negative pool; bootstrap resamples scored image pairs, not regenerated negative pools',interval='95% percentile, exploratory internal development set; no multiple-comparison correction; not cross-training-seed or independent-test inference',results=results))
    assert all(sha(ROOT/p)==h for p,h in before.items())
    assert all(sha(RUN/f'decoded/epoch-70/val/image_{int(i):06d}.npz')==h for i,h in old_hash.items())
    save(COMPARE/'completion.json',dict(training_updates_added=0,decode_exit_codes=codes,existing100_unchanged=True,protected_files_unchanged=True,epoch80_results_reused=True,all500_ids_gt_and_seeds_match=True,complete=True))
    (COMPARE/'COMPLETE').write_text('Evaluation-only completion; no training updates.\n')
    event('completed',training_updates_added=0)
if __name__=='__main__':
    try:main()
    except Exception as e:event('failed',error=repr(e));raise
