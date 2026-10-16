"""Four table identification features under Takagi's five-correlation averaging protocol.

No selection and no pixel average. Other four eight-table metrics are undefined
by this author protocol and must not be filled with K1 or invented best-of-five.
"""
import json
import os
from pathlib import Path
import sys

R=Path('/data1/matengyu/geyugong/neuroadapter-author-first')
B=R/'benchmarks/subj01-v1'
A=R/'reports/native-protocols-20261016/takagi'
O=B/'outputs/takagi-native5'
os.environ.update(TORCH_HOME=str(R/'runtime/torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
sys.path.insert(0,str(R/'scripts'))
from takagi_native5 import save
from mindbridge_native8 import sha


def main():
    assert (A/'generation-complete.json').exists()
    assert not (A/'native-identification.json').exists()
    import numpy as np
    import torch
    import torchvision
    from torchvision import transforms
    from torchvision.models.feature_extraction import create_feature_extractor
    from transformers import CLIPModel,CLIPProcessor
    from PIL import Image
    torch.set_num_threads(4)
    ids=[int(x) for x in (B/'test_full_ids.txt').read_text().split()]
    paths=[]
    for i in ids:
        rec=json.loads((O/f'{i}.json').read_text());assert rec['nsd_id0']==i and rec['K']==5
        row=[B/'assets/gt'/f'{i}.png']
        for candidate in rec['candidates']:
            p=O/candidate['path'];assert sha(p)==candidate['sha256'];row.append(p)
        assert len(row)==6;paths.append(row)
    preprocess=transforms.Compose([transforms.Resize(299),transforms.CenterCrop(299),transforms.ToTensor(),
               transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    inception=create_feature_extractor(torchvision.models.inception_v3(weights='IMAGENET1K_V1').eval(),{'flatten':'Incep'}).cuda()
    alex=create_feature_extractor(torchvision.models.alexnet(weights='IMAGENET1K_V1').eval(),{'features.5':'AN2','features.12':'AN5'}).cuda()
    clip=CLIPModel.from_pretrained(str(B/'assets/clip-vit-large-patch14'),local_files_only=True).eval().cuda()
    processor=CLIPProcessor.from_pretrained(str(B/'assets/clip-vit-large-patch14'),local_files_only=True)
    feature_dir=A/'native-features';feature_dir.mkdir(exist_ok=False)
    matrices={}
    with torch.no_grad():
        for index,row in enumerate(paths):
            for k,path in enumerate(row):
                image=Image.open(path).convert('RGB')
                batch=preprocess(image).unsqueeze(0).cuda()
                features={**inception(batch),**alex(batch)}
                inputs=processor(text='',images=image,return_tensors='pt').to('cuda')
                features['CLIP']=clip(**inputs,output_hidden_states=True).image_embeds
                for name,feature in features.items():
                    array=feature.flatten().cpu().numpy()
                    assert np.isfinite(array).all()
                    if name not in matrices:
                        matrices[name]=np.lib.format.open_memmap(feature_dir/f'{name}.npy',mode='w+',dtype=np.float32,shape=(982,6,len(array)))
                    matrices[name][index,k]=array
            save(A/'native-feature-progress.json',dict(completed=index+1,n=982))
            print(json.dumps(dict(features_completed=index+1)),flush=True)
    for matrix in matrices.values():matrix.flush()
    del inception,alex,clip
    torch.cuda.empty_cache()
    results={};detailed={'nsd_id0':ids};equivalence={}
    for name,raw in matrices.items():
        # Pearson correlation = dot of centered unit-length features. Averaging
        # normalized candidate vectors implements the author's mean of five r's.
        gt=np.asarray(raw[:,0],dtype=np.float64)
        gt-=gt.mean(axis=1,keepdims=True);gt/=np.linalg.norm(gt,axis=1,keepdims=True)
        averaged=np.zeros_like(gt)
        for k in range(1,6):
            candidate=np.asarray(raw[:,k],dtype=np.float64)
            candidate-=candidate.mean(axis=1,keepdims=True);candidate/=np.linalg.norm(candidate,axis=1,keepdims=True)
            averaged+=candidate/5
        similarity=np.clip(gt@averaged.T,-1,1)
        errors=[]
        for x,y in [(0,0),(0,1),(1,0),(1,1),(981,981),(14,238)]:
            reference=np.corrcoef(raw[x,0],raw[y,1:])[0,1:].mean()
            errors.append(abs(float(similarity[x,y])-float(reference)))
        assert max(errors)<1e-10
        accuracy=((np.diag(similarity)[:,None]>similarity).sum(axis=1))/981
        # Self comparisons must always be false; preserve the author's strict >.
        assert np.isfinite(accuracy).all() and np.all((accuracy>=0)&(accuracy<=1))
        detailed[name]=accuracy*100;results[name]=float(accuracy.mean()*100)
        equivalence[name]=max(errors)
    import pandas as pd
    pd.DataFrame(detailed).to_csv(A/'native-identification-per-image.csv',index=False)
    sources=B/'upstream/takagi/codes/utils'
    save(A/'native-identification.json',dict(n=982,K=5,negative_pool=982,mean=results,
         selection='none',aggregation='Mean of five Pearson correlations BEFORE strict pairwise comparison',
         original_formula_error=equivalence,feature_input='Original PIL Resize299/CenterCrop299 for AlexNet and Inception; original HF CLIPProcessor',
         source_sha256={f:sha(sources/f) for f in ['img2feat_decoded.py','identification.py']},
         undefined_table_metrics=['PixCorr','SSIM','Eff','SwAV'],
         feature_files={n:sha(feature_dir/f'{n}.npy') for n in matrices}))


if __name__=='__main__':main()
