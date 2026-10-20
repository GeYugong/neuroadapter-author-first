"""Verify final metrics and render fixed32 raw-tensor comparison/candidate atlases."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import h5py
from PIL import Image,ImageDraw,ImageFont
from evaluate_mindeye_subject02 import R,O,IDS,frozen,hashes,sha,write

plan=frozen()
metrics=json.loads((O/'metrics.json').read_text())
ids=np.loadtxt(IDS,dtype=int).tolist()
assert metrics['n']==metrics['negative_pool']==982
assert {int(x.stem) for x in (O/'records').glob('*.json')}==set(ids)
assert len(list((O/'outputs').glob('*.npz')))==982
counts={i:0 for i in range(16)}
manifest_hash=hashlib.sha256()
for i in ids:
    path=O/'records'/f'{i}.json';rec=json.loads(path.read_text())
    manifest_hash.update(f'{i}:{sha(path)}\n'.encode())
    assert rec['image_id']==i and len(rec['scores'])==len(rec['candidate_tensor_sha256'])==16
    assert np.isfinite(rec['scores']).all() and rec['pick']==int(np.argmax(rec['scores']))
    assert rec['seed']==42 and not rec['GT_used_for_selection']
    output=O/'outputs'/rec['output_file'];assert sha(output)==rec['output_sha256']
    counts[rec['pick']]+=1
    with np.load(output) as data:
        assert int(data['image_id'])==i
        for name,index in [('k1',0),('selected',rec['pick'])]:
            a=data[name];assert a.dtype==np.float32 and a.shape==(3,512,512) and np.isfinite(a).all()
            assert a.min()>=0 and a.max()<=1
            assert hashes([a])[0]==rec['candidate_tensor_sha256'][index]
        if i in ids[:32]:
            assert data['all_candidates'].shape==(16,3,512,512)
            assert hashes(data['all_candidates'])==rec['candidate_tensor_sha256']
            assert np.array_equal(data['k1'],data['all_candidates'][0])
            assert np.array_equal(data['selected'],data['all_candidates'][rec['pick']])
for label in ['K1','native16']:
    frame=pd.read_csv(O/f'{label}-per-image.csv')
    assert frame['image_id'].tolist()==ids and np.isfinite(frame.to_numpy()).all()
    for key,value in metrics['metrics'][label].items():assert abs(frame[key].mean()-value)<1e-10
for name in ['preflight-0','preflight-1','generate-0','generate-1','score-0']:
    assert json.loads((O/f'{name}-exit.json').read_text())['returncode']==0
write(O/'final-audit.json',dict(complete=True,n=982,candidate_records=982*16,
    original_float_selected_and_K1_verified=True,fixed32_all_candidates_verified=512,
    all_scores_argmax_IDs_and_hashes_verified=True,per_image_means_verified=True,
    negative_pool=982,selected_index_counts=counts,records_ordered_digest=manifest_hash.hexdigest(),
    weight_selection_unchanged=True,visual_review_pending=True,time=time.time()))

target=O/'review-atlas';target.mkdir(exist_ok=False)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
cfg=json.loads((R/'reports/subject02-bf16-seed42/effective-config.json').read_text())
pages=[]
def to_image(a,size):
    if a.ndim==3 and a.shape[0]==3:a=(a.transpose(1,2,0).clip(0,1)*255).astype(np.uint8)
    return Image.fromarray(a).resize((size,size),Image.Resampling.LANCZOS)
with h5py.File(Path(cfg['imgs_dir'])/'nsd_stimuli.hdf5','r') as gt:
    for page in range(4):
        canvas=Image.new('RGB',(872,2564),'white');draw=ImageDraw.Draw(canvas)
        draw.text((16,8),f'MindEye1 Subject2 / fixed IDs {8*page+1}-{8*page+8}',fill='black',font=title)
        for col,label in enumerate(['GT','Candidate0 (K1)','Native16 selected']):draw.text((16+280*col,48),label,fill='black',font=font)
        for row,i in enumerate(ids[page*8:page*8+8]):
            rec=json.loads((O/'records'/f'{i}.json').read_text())
            with np.load(O/'outputs'/rec['output_file']) as data:
                arrays=[gt['imgBrick'][i],data['k1'],data['selected']]
                for col,a in enumerate(arrays):
                    x,y=16+280*col,80+308*row
                    draw.text((x,y),f'ID {i}' if col==0 else ('c0' if col==1 else f'c{rec["pick"]} / s={max(rec["scores"]):.4f}'),fill='black',font=font)
                    canvas.paste(to_image(a,264),(x,y+26))
        path=target/f'comparison-{page+1}.png';canvas.save(path);pages.append({'file':path.name,'sha256':sha(path)})
    for page in range(8):
        canvas=Image.new('RGB',(1832,2004),'white');draw=ImageDraw.Draw(canvas)
        draw.text((16,8),f'All16 candidates / fixed IDs {4*page+1}-{4*page+4} / green=selected',fill='black',font=title)
        for row,i in enumerate(ids[page*4:page*4+4]):
            rec=json.loads((O/'records'/f'{i}.json').read_text())
            with np.load(O/'outputs'/rec['output_file']) as data:
                for half in range(2):
                    y=70+(row*2+half)*238
                    draw.text((16,y),f'GT {i}',fill='black',font=font)
                    canvas.paste(to_image(gt['imgBrick'][i],188),(16,y+26))
                    for col in range(8):
                        c=8*half+col;x=16+(col+1)*200
                        draw.text((x,y),f'c{c} / s={rec["scores"][c]:.4f}',fill='black',font=font)
                        canvas.paste(to_image(data['all_candidates'][c],188),(x,y+26))
                        if c==rec['pick']:draw.rectangle((x-2,y+24,x+189,y+215),outline='#07834b',width=3)
        path=target/f'candidates-{page+1}.png';canvas.save(path);pages.append({'file':path.name,'sha256':sha(path)})
write(O/'atlas-manifest.json',dict(ids=ids[:32],pages=pages,rule='Predeclared first32 standard Subject2 IDs; no reselection',
    display_only='Original tensors remain unchanged; thumbnails uniformly resized, no cropping or enhancement',time=time.time()))
print('FINAL_AUDIT_AND_ATLASES_READY_NEED_ACTUAL_VISUAL_REVIEW',flush=True)
