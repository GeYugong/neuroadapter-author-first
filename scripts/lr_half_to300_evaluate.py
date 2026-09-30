"""Fixed post-training val500 evaluation; reuse author decode and metric functions."""
import argparse, datetime, fcntl, hashlib, json, os, subprocess, sys
from pathlib import Path
import evaluate_stage as ev
import lr_half_evaluate as original
ROOT=ev.ROOT
RUN=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
REPORT=ROOT/'reports/lr-half-to300'
def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2)
def phase(epoch,mode):
    import numpy as np
    base=REPORT/f'epoch-{epoch}'; out=base/'decoded'; ids=ROOT/'configs/val_ids.txt'
    if mode=='decode':
        assert not base.exists(), 'Preserve prior artifacts; explicit recovery required'
        base.mkdir()
        ev.decode(RUN,epoch,'val',ids,out)
    elif mode=='score':
        assert not (base/'val500').exists()
        identity=original.identity(RUN,epoch,epoch*532,out)
        manifest=[]
        for i in np.loadtxt(ids,dtype=int).reshape(-1):
            p=out/f'image_{i:06d}.npz'
            original.verify_file(p,int(i),ROOT/f'runs/subject01-author-fp32-seed42/decoded/epoch-100/val/{p.name}')
            with np.load(p) as z:
                assert z['candidate_images'].shape[0]==2 and np.isfinite(z['candidate_images']).all()
            manifest.append({'image_id':int(i),'sha256':sha(p)})
        assert len(manifest)==500 and len(list(out.glob('image_*.npz')))==500
        save(base/'generation-audit.json',{'identity':identity,'manifest':manifest,'n':500,'candidates_per_image':2})
        ev.metrics(out,ids,base/'val500')
        ev.atlas(out,ROOT/'configs/atlas_val32_ids.txt',base/'val-atlas')
        save(base/'scoring-complete.json',{'epoch':epoch,'n':500,'negative_pool':500,'candidate_selection':False,'ai_review_pending':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    else: raise ValueError(mode)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epoch',type=int,choices=[200,250,300]); ap.add_argument('--phase',choices=['decode','score']); a=ap.parse_args()
    if a.phase:
        assert a.epoch is not None
        phase(a.epoch,a.phase);return
    lock=(ROOT/'runtime/lr-half-to300-eval.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit=read(REPORT/'training-final-audit.json'); assert audit['completed_epochs']==300 and audit['optimizer_updates']==159600 and audit['skipped_updates']==0
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for epoch in [200,250,300]:
        assert sha(RUN/f'exports/epoch-{epoch}/inference.pt')==audit['milestones'][str(epoch)]['export_sha256']
        for mode in ['decode','score']:
            tag=f'lr-half-to300-eval-{epoch}-{mode}'; log=ROOT/'logs'/f'{tag}.log'
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
            with log.open('x') as f:
                rc=subprocess.run([sys.executable,__file__,'--epoch',str(epoch),'--phase',mode],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            (log.with_suffix('.exit')).write_text(str(rc)+'\n')
            print(epoch,mode,'exit',rc,flush=True)
            if rc: raise SystemExit(rc)
    for p,h in read(REPORT/'validation-protected-before.json').items(): assert sha(ROOT/p)==h,p
    save(REPORT/'validation-scoring-complete.json',{'epochs':[200,250,300],'all_generation_scoring_exit0':True,'old_reports_unchanged':True,'ai_review_and_selection_pending':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
if __name__=='__main__': main()
