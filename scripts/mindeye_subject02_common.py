"""Pinned author components and isolated Subject 2 data helpers."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import textwrap
import types

R=Path('/data1/matengyu/geyugong/neuroadapter-author-first')
O=R/'reports/mindeye-subject02-20261019'
D=R/'data/mindeye-subject02'
S=R/'benchmarks/subj01-v1/upstream/mindeye1/src'
COMPAT=R/'runtime/mindeye-subject02-compat'
sys.path.insert(0,str(COMPAT))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_HOME=str(R/'runtime/torch'),
    MPLCONFIGDIR=str(R/'runtime/mindeye-subject02-mpl'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
    CUBLAS_WORKSPACE_CONFIG=':4096:8')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,obj,replace=False):
    path=Path(path);assert path.resolve().is_relative_to(R) and (replace or not path.exists())
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def source_components():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import kornia
    from kornia.augmentation.container import AugmentationSequential
    import huggingface_hub
    if not hasattr(huggingface_hub,'cached_download'):
        def unavailable_download(*args,**kwargs):
            raise RuntimeError('Legacy online cached_download disabled; use locked local assets')
        huggingface_hub.cached_download=unavailable_download
    source=(S/'models.py').read_text()
    replacements={
      'from_pretrained("openai/clip-vit-large-patch14")':f'from_pretrained({str(R/"benchmarks/subj01-v1/assets/clip-vit-large-patch14")!r}, attn_implementation="eager")',
      'clip.load(clip_variant, device=device)':f'clip.load({str(R.parent/"neuroadapter-subject1-research/models/evaluation/clip/ViT-L-14.pt")!r}, device="cpu")',
      'interpolation=transforms.InterpolationMode.BICUBIC),':'interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),',
    }
    for old,new in replacements.items():
        assert old in source,old
        source=source.replace(old,new)
    models=types.ModuleType('mindeye_subject02_models');models.__file__=str(S/'models.py')
    exec(compile(source,models.__file__,'exec'),models.__dict__)
    names={'seed_everything','mixco','mixco_nce','soft_clip_loss','soft_cont_loss','voxel_select','cosine_anneal','check_loss'}
    tree=ast.parse((S/'utils.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    ns=dict(torch=torch,np=np,random=random,os=os,F=F)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(S/'utils.py'),'exec'),ns)
    utils=SimpleUtils=types.SimpleNamespace(**{n:ns[n] for n in names})
    aug={}
    for mode,file,target in [('high','Train_MindEye.py','img_augment'),('low','train_autoencoder.py','train_augs')]:
        tree=ast.parse((S/file).read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets))
        scope=dict(kornia=kornia,AugmentationSequential=AugmentationSequential)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(S/file),'exec'),scope)
        aug[mode]=scope[target]
    return models,utils,aug,replacements


def loss_function(mode,model,frozen,utils,augment,epochs):
    """Compile the actual author's loss block, with no rewritten loss formula."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import kornia
    scope=dict(torch=torch,nn=nn,F=F,kornia=kornia,utils=utils,device=torch.device('cuda'),
               distributed=False,num_epochs=epochs,epoch=0,mixup_pct=.33,prior=True,v2c=True,
               hidden=True,clip_size=768,prior_mult=30,use_image_aug=True,img_augment=augment,
               soft_loss_temps=utils.cosine_anneal(.004,.0075,epochs-int(.33*epochs)))
    if mode=='high':
        text=(S/'Train_MindEye.py').read_text()
        start=text.index('            repeat_index = train_i % 3')
        end=text.index('            utils.check_loss(loss)',start)+len('            utils.check_loss(loss)')
        body=textwrap.dedent(text[start:end])
        scope.update(diffusion_prior=model,clip_extractor=frozen['clip'])
        src='def compute(voxel,image,epoch,train_i):\n    loss_nce_sum=0.;loss_prior_sum=0.\n'+textwrap.indent(body,'    ')+'\n    return loss,{"nce":loss_nce.detach(),"prior":loss_prior.detach()}\n'
    else:
        text=(S/'train_autoencoder.py').read_text()
        start=text.index('        image = image.to(device).float()',text.index('for epoch in progress_bar:'))
        end=text.index('            loss = mse_loss/0.18215 + 2*reconst_loss + 0.1*cont_loss + 16*sobel_loss',start)
        end+=len('            loss = mse_loss/0.18215 + 2*reconst_loss + 0.1*cont_loss + 16*sobel_loss')
        body=textwrap.dedent(text[start:end])
        scope.update(voxel2sd=model,autoenc=frozen['vae'],cnx=frozen['cnx'],train_augs=augment,
                     use_mp=False,use_cont=True,use_reconst=False,use_sobel_loss=False,
                     use_blurred_training=False,mixup_pct=-1,
                     mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None],
                     std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None])
        src='def compute(voxel,image,epoch,train_i):\n'+textwrap.indent(body,'    ')+'\n    return loss,{"latent_l1":mse_loss.detach(),"contrastive":cont_loss.detach()}\n'
    exec(compile(src,str(S/('Train_MindEye.py' if mode=='high' else 'train_autoencoder.py')),'exec'),scope)
    return scope['compute'],hashlib.sha256(body.encode()).hexdigest()


def index_tar(path):
    import numpy as np
    from PIL import Image
    rows={}
    with tarfile.open(path,'r:') as tar:
        for member in tar:
            suffix=member.name.split('.',1)[-1]
            if suffix not in ['jpg','nsdgeneral.npy','coco73k.npy']:continue
            assert member.isfile() and Path(member.name).name==member.name
            key=member.name.split('.',1)[0];row=rows.setdefault(key,{'key':key,'tar':str(path.relative_to(R)),'members':{}})
            data=tar.extractfile(member).read()
            row['members'][suffix]={'offset':member.offset_data,'bytes':member.size,'sha256':hashlib.sha256(data).hexdigest()}
            if suffix=='nsdgeneral.npy':
                a=np.load(io.BytesIO(data),allow_pickle=False)
                assert a.shape==(3,14278) and a.dtype==np.float16 and np.isfinite(a).all()
            elif suffix=='coco73k.npy':row['image_id']=int(np.load(io.BytesIO(data),allow_pickle=False).item())
            else:
                image=Image.open(io.BytesIO(data));assert image.mode=='RGB' and image.size==(256,256)
    assert all(len(x['members'])==3 and 'image_id' in x for x in rows.values())
    return list(rows.values())


def sample(row):
    import numpy as np
    import torch
    from PIL import Image
    values={}
    with (R/row['tar']).open('rb') as f:
        for suffix in ['jpg','nsdgeneral.npy']:
            m=row['members'][suffix];f.seek(m['offset']);data=f.read(m['bytes'])
            assert hashlib.sha256(data).hexdigest()==m['sha256']
            values[suffix]=data
    brain=torch.from_numpy(np.load(io.BytesIO(values['nsdgeneral.npy']),allow_pickle=False).copy())
    img=torch.from_numpy(np.asarray(Image.open(io.BytesIO(values['jpg']))).copy()).permute(2,0,1).float()/255
    return brain,img,row['image_id']
