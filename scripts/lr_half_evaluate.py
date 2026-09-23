"""Isolated B0/H/L evaluation; reuse original decode, metrics and atlas functions."""
import argparse,hashlib,json,subprocess,sys,fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import evaluate_stage as ev
ROOT=ev.ROOT;MAIN=ROOT/'runs/subject01-author-fp32-seed42';L=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
REPORT=ROOT/'reports/lr-half-from100';WORK=L/'comparison';CFG=ROOT/'configs'
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def ids(name):return np.loadtxt(CFG/name,dtype=int).reshape(-1).tolist()
def identity(run,epoch,updates,folder):
    export=run/f'exports/epoch-{epoch}/inference.pt';m=read(export.with_name('metadata.json'))
    assert (m['epoch'],m['global_optimizer_updates'],m['sha256'])==(epoch,updates,sha(export))
    p=read(folder/'protocol.json');assert Path(p['export_path'])==export
    for k,v in dict(candidate_batch=1,num_predictions=2,denoising_steps=50,cfg=4,seed=42,selection=False,gt_initialization=False,weight_dtype='torch.float32').items():assert p[k]==v
    return {'metadata':m,'protocol':p,'sha256':sha(export)}
def verify_file(p,i,ref=None):
    with np.load(p) as z:
        assert int(z['image_id'])==i and len(z['candidate_images'])==2
        assert list(z['seeds'])==[int.from_bytes(hashlib.sha256(f'42:{i}:{k}'.encode()).digest()[:8],'little')%(2**63-1) for k in range(2)]
        if ref:
            with np.load(ref) as b:assert np.array_equal(z['groundtruth_image'],b['groundtruth_image']) and np.array_equal(z['seeds'],b['seeds'])
def phase(name):
    full=ids('val_ids.txt');monitor=ids('monitor100_ids.txt');h=MAIN/'decoded/epoch-110/val';b=MAIN/'decoded/epoch-100/val'
    if name=='prepare':
        assert not WORK.exists();WORK.mkdir()
        provenance={'B0':identity(MAIN,100,53200,b),'H':identity(MAIN,110,58520,h)}
        for label,epoch in [('B0',100),('H',110)]:
            audit=read(ROOT/f'reports/epoch-{epoch}/delivery-checks.json')
            if label=='H':assert audit['hashes'][f'runs/subject01-author-fp32-seed42/exports/epoch-{epoch}']['inference.pt']==provenance[label]['sha256']
            assert (ROOT/f'reports/epoch-{epoch}/COMPLETE').exists()
        existing={}
        for i in full:verify_file(b/f'image_{i:06d}.npz',i)
        for i in monitor:
            p=h/f'image_{i:06d}.npz';verify_file(p,i,b/p.name);existing[str(i)]=sha(p)
        provenance['H_existing100_sha256']=existing
        provenance['B0_metrics_sha256']=sha(ROOT/'reports/epoch-100/val500/metrics.json')
        save(REPORT/'evaluation-provenance.json',provenance)
        np.savetxt(WORK/'H-missing400.txt',[i for i in full if i not in set(monitor)],fmt='%d')
    elif name=='H-generate':
        assert not (WORK/'H-missing400').exists()
        ev.decode(MAIN,110,'val',WORK/'H-missing400.txt',WORK/'H-missing400')
        union=WORK/'H-full500';union.mkdir();manifest=[];old=read(REPORT/'evaluation-provenance.json')['H_existing100_sha256']
        for i in full:
            src=(h if i in monitor else WORK/'H-missing400')/f'image_{i:06d}.npz';verify_file(src,i,b/src.name)
            if i in monitor:assert sha(src)==old[str(i)]
            (union/src.name).symlink_to(src);manifest.append(dict(image_id=i,source=str(src),sha256=sha(src),reused=i in monitor))
        save(REPORT/'H-full500-manifest.json',manifest)
    elif name=='H-score':
        ev.metrics(WORK/'H-full500',CFG/'val_ids.txt',REPORT/'H/val500')
        ev.metrics(WORK/'H-full500',CFG/'monitor100_ids.txt',REPORT/'H/val100')
    elif name=='L-generate':
        state=read(L/'checkpoint-110/progress.json');assert state['completed_epochs']==110 and state['global_optimizer_updates']==58520 and state['current_lr']==[5e-5]
        assert not (L/'decoded/epoch-110').exists()
        ev.decode(L,110,'val',CFG/'val_ids.txt',L/'decoded/epoch-110/val')
        ev.decode(L,110,'train',CFG/'atlas_train16_ids.txt',L/'decoded/epoch-110/train')
    elif name=='L-score':
        identity(L,110,58520,L/'decoded/epoch-110/val')
        for i in full:verify_file(L/f'decoded/epoch-110/val/image_{i:06d}.npz',i,b/f'image_{i:06d}.npz')
        ev.metrics(L/'decoded/epoch-110/val',CFG/'val_ids.txt',REPORT/'L/val500')
        ev.metrics(L/'decoded/epoch-110/val',CFG/'monitor100_ids.txt',REPORT/'L/val100')
        ev.metrics(L/'decoded/epoch-110/train',CFG/'atlas_train16_ids.txt',REPORT/'L/train16')
        ev.atlas(L/'decoded/epoch-110/val',CFG/'atlas_val32_ids.txt',REPORT/'L/val-atlas')
        ev.atlas(L/'decoded/epoch-110/train',CFG/'atlas_train16_ids.txt',REPORT/'L/train-atlas')
    elif name=='compare':
        keys=['PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV'];frames={};summaries={}
        for label,base in [('B0',ROOT/'reports/epoch-100'),('H',REPORT/'H'),('L',REPORT/'L')]:
            cs=[pd.read_csv(base/f'val500/candidate-{k}.csv').set_index('image_id').loc[full,keys] for k in range(2)]
            frames[label]=(cs[0]+cs[1])/2;frames[label]['S']=frames[label][['Alex(5)','Incep','CLIP']].mean(axis=1)
            met=read(base/'val500/metrics.json');assert np.allclose(frames[label].mean(),[met['mean'][k] for k in keys]+[met['S']],atol=1e-10,rtol=0)
            summaries[label]={pool:read(base/pool/'metrics.json') for pool in ['val100','val500']}
        results={};rng=np.random.default_rng(20260923);draws=rng.integers(0,500,size=(10000,500))
        for ref in ['B0','H']:
            d=frames['L']-frames[ref];d.to_csv(REPORT/f'paired-L-minus-{ref}.csv');values=d.to_numpy();samples=np.empty((10000,9))
            for start in range(0,10000,100):samples[start:start+100]=values[draws[start:start+100]].mean(axis=1)
            rows=[]
            for j,k in enumerate(keys+['S']):
                q=[.0125,.9875] if k=='S' else [.025,.975];ci=np.quantile(samples[:,j],q)
                rows.append(dict(metric=k,delta=float(values[:,j].mean()),interval_level=.975 if k=='S' else .95,low=float(ci[0]),high=float(ci[1]),better='lower' if k in ['Eff','SwAV'] else 'higher'))
            results[ref]=rows
        save(REPORT/'comparison.json',dict(seed=20260923,resamples=10000,sampling_unit='image after averaging two candidates; fixed500 metric pool',scope='Exploratory repeatedly used internal development set; not independent test',summaries=summaries,contrasts=results))
        for p,digest in read(REPORT/'protected-before.json').items():assert sha(ROOT/p)==digest, 'Protected mainline changed: '+p
        (REPORT/'EVALUATION_COMPLETE').write_text('B0/H/L scoring complete; AI atlas review and final delivery still required.\n')
    (REPORT/f'{name}.done').write_text('exit0\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase');args=parser.parse_args()
    if args.phase:phase(args.phase);return
    lock=(ROOT/'runtime/lr-half-eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import stage3 as common
    def event(**kw):
        with (ROOT/'logs/lr-half-eval-events.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
    common.event=event
    assert (L/'checkpoint-110/COMPLETE').exists()
    for name in ['prepare','H-generate','H-score','L-generate','L-score','compare']:
        if (REPORT/f'{name}.done').exists():continue
        common.execute('lr-half-eval-'+name,[sys.executable,__file__,'--phase',name])
if __name__=='__main__':main()
