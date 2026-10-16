"""Restore the pinned Takagi author's five iterations, without inventing a selector."""
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

R=Path('/data1/matengyu/geyugong/neuroadapter-author-first')
B=R/'benchmarks/subj01-v1'
A=R/'reports/native-protocols-20261016/takagi'
O=B/'outputs/takagi-native5'
sys.path.insert(0,str(B))
import takagi_reconstruct_k1 as old


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def code():
    assert old.sha(old.SRC)==old.SOURCE_SHA
    tree=ast.parse(old.SRC.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('load_model_from_config','load_img_from_arr')]
    # PyTorch 2.6+ defaults changed; this trusted pinned checkpoint needs its original loader behavior.
    for node in ast.walk(functions[0]):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.load':
            node.keywords.append(ast.keyword(arg='weights_only',value=ast.Constant(False)))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    def target(n):
        return n.targets[0].id if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) else None
    start=next(i for i,n in enumerate(main.body) if target(n)=='n_samples')
    body=ast.Module(body=main.body[start:],type_ignores=[])
    counts={'outdir':0,'scores_latent':0,'scores_c':0,'GT_save':0}
    class Adapt(ast.NodeTransformer):
        def visit_Assign(self,n):
            name=target(n)
            mapping={'outdir':'_NATIVE_OUT','scores_latent':'_LATENTS','scores_c':'_CONDITIONS'}
            if name in mapping:
                n.value=ast.Name(id=mapping[name],ctx=ast.Load());counts[name]+=1
            return self.generic_visit(n)
        def visit_Expr(self,n):
            if 'sdataset' in ast.unparse(n):
                assert '_org.png' in ast.unparse(n);counts['GT_save']+=1;return None
            return self.generic_visit(n)
    body=ast.fix_missing_locations(Adapt().visit(body))
    assert all(v==1 for v in counts.values())
    assert any(target(n)=='n_iter' and n.value.value==5 for n in body.body)
    functions=ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]))
    return compile(functions,str(old.SRC),'exec'),compile(body,str(old.SRC),'exec'),counts


def compatibility():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path[:0]=[str(R/'runtime/native-protocol-compat'),str(old.SD)]
    sys.path.extend([str(B/'envs/p1-author310/lib/python3.10/site-packages'),str(B/'envs/mindbridge/lib/python3.10/site-packages')])
    import pytorch_lightning.utilities.rank_zero as rank_zero
    sys.modules['pytorch_lightning.utilities.distributed']=rank_zero


def prepare():
    _,native,ids,latent,condition,plan=old.preflight('testfull')
    compiled=code()
    compatibility()
    from ldm.models.diffusion.ddpm import LatentDiffusion
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    import torch
    config=OmegaConf.load(old.SD/'configs/stable-diffusion/v1-inference.yaml')
    config.model.params.cond_stage_config.params={'version':str(B/'assets/clip-vit-large-patch14')}
    model=instantiate_from_config(config.model)
    checkpoint=torch.load(old.MODEL,map_location='cpu',weights_only=False)
    missing,unexpected=model.load_state_dict(checkpoint['state_dict'],strict=False)
    # Author allows unused EMA/training state; all actual inference parameters must be loaded.
    assert not missing, missing
    save(A/'preflight.json',dict(n=982,K=5,selection='none; average candidate feature correlations before pairwise identification',
         native_parameters=dict(ddim_steps=50,strength=.8,cfg=5,eta=0,seed=42),
         source_sha256=old.SOURCE_SHA,checkpoint_sha256=old.MODEL_SHA,
         ast_adaptations=compiled[2],input_audit=plan,missing_keys=missing,unexpected_keys=unexpected,
         runtime_compatibility='Python3.11 runtime; Lightning2.5.5 base module and relocated rank_zero_only; original diffusion code',
         gpu_forward_verified=False,metric_note='Original paper identification is not best-of-five; unspecified eight-table metrics remain separate'))


def generate():
    assert (A/'preflight.json').exists() and not (A/'generation-started.json').exists()
    from mindbridge_native8 import free_gpu
    assert free_gpu(os.environ.get('CUDA_VISIBLE_DEVICES'))==os.environ.get('CUDA_VISIBLE_DEVICES')
    assert shutil.disk_usage(R).free>15*2**30
    _,native,ids,latent,condition,plan=old.preflight('testfull')
    compiled=code();compatibility()
    import numpy as np
    import torch
    import PIL
    from PIL import Image
    from omegaconf import OmegaConf
    from einops import rearrange
    from torch import autocast
    from contextlib import nullcontext
    from tqdm import trange
    from pytorch_lightning import seed_everything
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    O.mkdir(parents=True,exist_ok=True);assert not list(O.iterdir())
    save(A/'generation-started.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],runner_sha256=old.sha(Path(__file__))))
    namespace=dict(torch=torch,np=np,PIL=PIL,Image=Image,os=os,rearrange=rearrange,autocast=autocast,
                   nullcontext=nullcontext,trange=trange,instantiate_from_config=instantiate_from_config,DDIMSampler=DDIMSampler)
    exec(compiled[0],namespace)
    seed_everything(42)
    config=OmegaConf.load(old.SD/'configs/stable-diffusion/v1-inference.yaml')
    config.model.params.cond_stage_config.params={'version':str(B/'assets/clip-vit-large-patch14')}
    torch.cuda.set_device(0)
    model=namespace['load_model_from_config'](config,str(old.MODEL),0)
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def run(image_id,folder):
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        row=int(np.flatnonzero(native==image_id)[0])
        scope=dict(namespace,model=model,gpu=0,method='cvpr',subject='subj01',imgidx=row,stims_ave=native,
                   tr_idx=np.zeros_like(native),_LATENTS=latent,_CONDITIONS=condition,_NATIVE_OUT=str(folder))
        exec(compiled[1],scope)
        return [folder/'samples'/f'{row:05}_{k:03}.png' for k in range(5)]
    save(O/'protocol.json',json.loads((A/'preflight.json').read_text()))
    for index,value in enumerate(ids):
        image_id=int(value)
        paths=run(image_id,O/str(image_id))
        assert all(p.exists() for p in paths)
        if index==0:
            # Cache/RNG check uses the fixed first ID only; it does not select candidates.
            reference=run(image_id,A/'cache-identity')
            assert all(np.array_equal(np.array(Image.open(p)),np.array(Image.open(q))) for p,q in zip(paths,reference))
            save(A/'gpu-cache-identity.json',dict(id=image_id,all_five_pixel_identical=True))
        save(O/f'{image_id}.json',dict(nsd_id0=image_id,K=5,seed=42,selection='none',
             candidates=[dict(path=str(p.relative_to(O)),sha256=old.sha(p)) for p in paths]))
        save(A/'generation-progress.json',dict(completed=index+1,n=982,last_id=image_id))
        print(json.dumps(dict(completed=index+1,nsd_id0=image_id)),flush=True)
    save(A/'generation-complete.json',dict(n=982,K=5))


def controller():
    assert (A/'preflight.json').exists() and not (A/'controller-started.json').exists()
    save(A/'controller-started.json',dict(pid=os.getpid(),runner_sha256=old.sha(Path(__file__))))
    from mindbridge_native8 import free_gpu
    while True:
        prior=R/'reports/native-protocols-20261016/mindbridge/controller-status.json'
        ready=prior.exists() and json.loads(prior.read_text()).get('status')=='metrics_complete_needs_review'
        gpu=free_gpu() if ready else None
        if gpu is not None: break
        save(A/'controller-status.json',dict(status='waiting_prior_metrics' if not ready else 'waiting_free_gpu',pid=os.getpid()));time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
    with (A/'generation.log').open('x') as log:
        result=subprocess.call([str(R/'runtime/env/bin/python'),'-B',str(Path(__file__).resolve()),'generate'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(A/'generation-exit.json',dict(exit_code=result))
    if result!=0:
        save(A/'controller-status.json',dict(status='failed',exit_code=result));return
    with (A/'native-score.log').open('x') as log:
        result=subprocess.call([str(R/'runtime/env/bin/python'),'-B',str(R/'scripts/score_takagi_native5.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(A/'score-exit.json',dict(exit_code=result))
    save(A/'controller-status.json',dict(status='metrics_complete_needs_review' if result==0 else 'score_failed',exit_code=result))


if __name__=='__main__':
    {'prepare':prepare,'generate':generate,'controller':controller}[sys.argv[1]]()
