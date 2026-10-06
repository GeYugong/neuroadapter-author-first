from pathlib import Path
import json, numpy as np, torch
from PIL import Image
R=Path('/data1/matengyu/geyugong/neuroadapter-author-first'); B=R/'benchmarks/subj01-v1'
torch.set_num_threads(2)
ids=np.loadtxt(B/'test_full_ids.txt',dtype=int).reshape(-1)
paths={'mindeye1':'mindeye1','brain-diffuser':'brain-diffuser-full982','takagi':'takagi'}
for method in ['L400','mindeye1','brain-diffuser','takagi','mindbridge']:
    lo=1.;hi=0.;bad=0;bad_pixels=0;bad_ids=[]
    for i in ids:
        if method=='L400':
            with np.load(R/f'reports/lr-half-to500/test982/decoded/image_{i:06d}.npz') as d:
                assert int(d['image_id'])==i and d['candidate_images'].shape[0]==1
                img=d['candidate_images'][0].copy()
            x=torch.from_numpy(img).permute(2,0,1).float()/255
        elif method=='mindbridge':
            native=torch.load(B/'outputs/mindbridge'/f'{i}.pt',map_location='cpu',weights_only=True)
            assert native.shape==(1,1,3,512,512) and native.dtype==torch.float32
            x=native[0,0]
        else:
            img=np.array(Image.open(B/'outputs'/paths[method]/f'{i}.png').convert('RGB'))
            x=torch.from_numpy(img).permute(2,0,1).float()/255
        y=torch.nn.functional.interpolate(x.unsqueeze(0),size=(425,425),mode='bilinear',align_corners=False)
        assert torch.isfinite(y).all()
        mi=float(y.min()); ma=float(y.max());lo=min(lo,mi);hi=max(hi,ma)
        n=int(((y<0)|(y>1)).sum());bad_pixels+=n
        if n:bad+=1;bad_ids.append(int(i))
    print(json.dumps(dict(method=method,n= len(ids),min=lo,max=hi,outside_images=bad,outside_pixels=bad_pixels,outside_ids=bad_ids)),flush=True)
