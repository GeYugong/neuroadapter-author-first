"""Fixed post-training val500 evaluation; reuse author decode and metric functions."""
import argparse, datetime, fcntl, hashlib, json, os, subprocess, sys
from pathlib import Path
import evaluate_stage as ev
import lr_half_evaluate as original
ROOT=ev.ROOT
RUN=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-from400'
REPORT=ROOT/'reports/balanced-lr-from400'
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
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--epoch',type=int,choices=[450,500],required=True)
    ap.add_argument('--phase',choices=['decode','score'],required=True)
    a=ap.parse_args()
    progress=read(RUN/f'checkpoint-{a.epoch}/progress.json')
    assert progress['completed_epochs']==a.epoch and progress['skipped_updates']==0
    phase(a.epoch,a.phase)