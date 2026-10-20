"""Locked MindEye1 Subject2 native16 reconstruction and full982 evaluation."""
import argparse
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
import types

sys.path.insert(0,str(Path(__file__).resolve().parent))
from mindeye_subject02_common import R,S,sha,write,index_tar,sample,source_components
import numpy as np

O=R/'reports/mindeye-subject02-test-20261020'
D=R/'data/mindeye-subject02-test'
TRAIN=R/'reports/mindeye-subject02-20261019'
IDS=R/'data/subject02/standard-test-ids.txt'
HASHES=[
 ('test_subj02_0.tar','cf8d23c9523ee230c853b122109b01fdb3b11a3a630f9fa6071f590778dbf762'),
 ('test_subj02_1.tar','dcde6832b25f67869e7103ffb2da7a12079a381802fbc2681eb3b24cfea3174b')]


def npz(path,**arrays):
    assert path.resolve().is_relative_to(O) and not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)


def prepare():
    import torch
    torch.set_num_threads(4)
    O.mkdir(exist_ok=True)
    rows=[]
    for name,digest in HASHES:
        path=D/name
        assert sha(path)==digest
        rows.extend(index_tar(path))
    ids=np.loadtxt(IDS,dtype=int).tolist()
    assert len(rows)==982 and len({x['image_id'] for x in rows})==982
    lookup={x['image_id']:x for x in rows};assert set(ids)==set(lookup)
    training=json.loads((TRAIN/'data-index.json').read_text())
    assert not set(ids)&{x['image_id'] for part in training.values() for x in part}
    brains=[]
    for i in ids:
        brain,_,actual=sample(lookup[i]);assert actual==i and brain.dtype==torch.float16
        brains.append(brain.mean(0).float().numpy())
    brain=np.stack(brains);assert brain.shape==(982,14278) and np.isfinite(brain).all()
    npz(O/'test-brain.npz',ids=np.asarray(ids),voxels=brain)
    write(O/'test-input-audit.json',dict(n=982,subject=2,IDs_match_independently_verified_S2_order=True,
        train_validation_disjoint=True,mean_precision='Author torch float16 repeat mean THEN float32',
        brain_sha256=sha(O/'test-brain.npz'),source_shards=dict(HASHES),rows=rows))
    done=json.loads((TRAIN/'training-completion-audit.json').read_text())
    protected={str(IDS.relative_to(R)):sha(IDS),str(Path(__file__).relative_to(R)):sha(Path(__file__)),
        'scripts/mindeye_subject02_common.py':sha(R/'scripts/mindeye_subject02_common.py'),
        'metric_brain_adapter.py':sha(R/'metric_brain_adapter.py'),
        'benchmarks/subj01-v1/metric_feature_batches.py':sha(R/'benchmarks/subj01-v1/metric_feature_batches.py'),
        str((S/'utils.py').relative_to(R)):sha(S/'utils.py'),str((S/'models.py').relative_to(R)):sha(S/'models.py')}
    prior_lock=json.loads((TRAIN/'frozen-plan.json').read_text())
    for name,digest in prior_lock['code_sha256'].items():
        assert sha(R/name)==digest,name
        protected[name]=digest
    for folder in ['versatile-diffusion','clip-vit-large-patch14']:
        for path in (R/'benchmarks/subj01-v1/assets'/folder).rglob('*'):
            if path.is_file() and path.suffix in ['.json','.bin','.safetensors']:
                protected[str(path.relative_to(R))]=sha(path)
    for kind in ['high','low']:
        path=R/f'runs/mindeye-subject02-{kind}/last.pth'
        assert sha(path)==done[kind]['final_export_sha256'];protected[str(path.relative_to(R))]=sha(path)
    write(O/'plan.json',dict(subject=2,n=982,K=16,seed=42,seed_policy='Native fresh CUDA generator seed42 per image; fixed16-candidate stream, no seed search',
        weights={'high_epoch':240,'low_epoch':120},primary='native16',paired_control='candidate0 of the SAME16',
        selection='Author cosine similarity of brain-predicted CLIP projector embedding and candidate CLIP features; no GT image',
        diffusion_steps=20,prior_steps=100,guidance_scale=3.5,img2img_strength=.85,
        negative_pool=982,feature_batch=16,protected_sha256=protected,
        test_brain_sha256=sha(O/'test-brain.npz'),fixed_visual_ids=ids[:32],
        storage='Lossless original float tensors for selected and candidate0 for all982. All16 candidate tensors for predeclared first32. All16 scores and tensor hashes for every ID.',
        disk_reserve_gib=25,train_or_weight_selection_allowed=False,
        limitations=['Single training seed per branch; independent8559/300split differs from original test-monitoring script',
        'Current runtime and original pure-Python dependencies differ from historical author environment',
        'MindEye uses official volumetric nsdgeneral inputs and native repeat averaging; NeuroAdapter uses surface parcels/full40-session averaging',
        'Native16 versus NeuroAdapter8 has a different candidate budget; report protocols explicitly']))
    print('TEST_INPUTS_READY',flush=True)


def frozen():
    plan=json.loads((O/'plan.json').read_text())
    for name,digest in plan['protected_sha256'].items():assert sha(R/name)==digest,name
    assert sha(O/'test-brain.npz')==plan['test_brain_sha256']
    assert shutil.disk_usage(R).free>25*1024**3
    return plan


def load():
    import torch
    torch.set_num_threads(4)
    models,utils,_,_=source_components()
    utils.seed_everything(42);torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    from diffusers import VersatileDiffusionDualGuidedPipeline,UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    low=models.Voxel2StableDiffusionModel(in_dim=14278,use_cont=True)
    brain=models.BrainNetwork(in_dim=14278,out_dim=257*768)
    net=models.VersatileDiffusionPriorNetwork(dim=768,depth=6,dim_head=64,heads=12,causal=False,learned_query_mode='pos_emb')
    prior=models.BrainDiffusionPrior(net=net,image_embed_dim=768,condition_on_text_encodings=False,
        timesteps=100,cond_drop_prob=.2,image_embed_scale=None,voxel2clip=brain)
    for kind,model,epoch in [('high',prior,240),('low',low,120)]:
        ckpt=torch.load(R/f'runs/mindeye-subject02-{kind}/last.pth',map_location='cpu',weights_only=True,mmap=True)
        assert ckpt['epoch']==epoch;model.load_state_dict(ckpt['model_state_dict'],strict=True)
        model.eval().requires_grad_(False).cuda()
    original_loop=prior.p_sample_loop
    def guarded(self,*a,**kw):
        assert kw.get('generator') is not None,'No-generator fallback forbidden'
        return original_loop(*a,**kw)
    prior.p_sample_loop=types.MethodType(guarded,prior)
    vdpath=R/'benchmarks/subj01-v1/assets/versatile-diffusion'
    vd=VersatileDiffusionDualGuidedPipeline.from_pretrained(str(vdpath),torch_dtype=torch.float16,local_files_only=True)
    count=0
    for module in vd.image_unet.modules():
        if isinstance(module,DualTransformer2DModel):
            count+=1;module.mix_ratio=0.
            module.condition_lengths[:]=[77,257];module.transformer_index_for_condition[:]=[1,0]
    assert count==16,count
    vd.remove_unused_weights()
    vd.scheduler=UniPCMultistepScheduler.from_pretrained(str(vdpath),subfolder='scheduler',local_files_only=True)
    vd.image_unet.eval().requires_grad_(False).cuda();vd.vae.eval().requires_grad_(False).cuda()
    clipper=models.Clipper('ViT-L/14',hidden_state=True,norm_embs=True,device=torch.device('cuda')).eval()
    source=(S/'utils.py').read_text();tree=ast.parse(source)
    names={'reconstruction','decode_latents','batchwise_cosine_similarity'}
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];assert len(nodes)==3
    ns=dict(torch=torch,np=np,nn=torch.nn,device=torch.device('cuda'),Clipper=models.Clipper)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(S/'utils.py'),'exec'),ns)
    reference=ns['reconstruction']
    node=next(x for x in nodes if x.name=='reconstruction')
    function=ast.get_source_segment(source,node)
    target='best_picks[0] = int(np.nanargmax(sims))'
    assert function.count(target)==2
    instrumented=function.replace(target,"globals()['last_sims'] = list(sims)\n        "+target,1)
    exec(compile(instrumented,str(S/'utils.py'),'exec'),ns)
    def reconstruct(voxel,record=True):
        utils.seed_everything(42)
        with torch.no_grad():
            voxel=torch.from_numpy(voxel[None]).cuda()
            blurry=vd.vae.decode(low(voxel.float()).half()/.18215).sample/2+.5
            fn=ns['reconstruction'] if record else reference
            _,images,picks,_=fn(torch.zeros(1,3,512,512),voxel,clipper,vd.image_unet,vd.vae,vd.scheduler,
                diffusion_priors=[prior],img_lowlevel=blurry,num_inference_steps=20,n_samples_save=1,
                recons_per_sample=16,guidance_scale=3.5,img2img_strength=.85,timesteps_prior=100,
                seed=42,retrieve=False,plotting=False,img_variations=False,verbose=False)
            assert images.shape==(1,16,3,512,512) and torch.isfinite(images).all()
            arr=images[0].float().cpu().numpy()
            assert arr.min()>=0 and arr.max()<=1
            scores=ns['last_sims'] if record else None
            if record:assert len(scores)==16 and np.isfinite(scores).all() and int(np.argmax(scores))==int(picks[0])
            return arr,int(picks[0]),scores
    return reconstruct


def hashes(arr):return [hashlib.sha256(x.tobytes()).hexdigest() for x in arr]


def save_result(i,arr,pick,scores,plan):
    name=f'image_{i:06d}.npz'
    payload=dict(image_id=i,k1=arr[0],selected=arr[pick])
    if i in plan['fixed_visual_ids']:payload['all_candidates']=arr
    npz(O/'outputs'/name,**payload)
    write(O/'records'/f'{i}.json',dict(image_id=i,K=16,seed=42,pick=pick,scores=scores,
        candidate_tensor_sha256=hashes(arr),output_file=name,output_sha256=sha(O/'outputs'/name),
        all_candidates_retained=i in plan['fixed_visual_ids'],GT_used_for_selection=False))


def generate(rank,preflight=False):
    plan=frozen();data=np.load(O/'test-brain.npz');ids=data['ids'].tolist();brain=data['voxels']
    fn=load();started=time.time()
    if preflight:
        a,p,_=fn(brain[0],False);b,q,scores=fn(brain[0],True)
        assert p==q and np.array_equal(a,b),'Instrumented versus original native reconstruction mismatch'
        npz(O/f'preflight-{rank}.npz',candidates=b)
        write(O/f'preflight-{rank}.json',dict(passed=True,image_id=ids[0],pick=q,scores=scores,
            candidate_tensor_sha256=hashes(b),original_native_exact=True,elapsed=time.time()-started))
        return
    for position in range(rank,len(ids),2):
        i=int(ids[position]);record=O/'records'/f'{i}.json'
        if record.exists():
            old=json.loads(record.read_text());assert sha(O/'outputs'/old['output_file'])==old['output_sha256'];continue
        assert shutil.disk_usage(R).free>25*1024**3
        arr,pick,scores=fn(brain[position]);save_result(i,arr,pick,scores,plan)
        write(O/f'progress-{rank}.json',dict(rank=rank,last_position=position,image_id=i,elapsed=time.time()-started),True)
        print('GENERATED',rank,position+1,i,pick,flush=True)


def score():
    import torch,h5py,pandas as pd
    torch.set_num_threads(4);plan=frozen()
    sys.path.insert(0,str(R/'benchmarks/subj01-v1'))
    import evaluate_stage
    import metric_brain_adapter as m
    from metric_feature_batches import install
    install(m,16);ids=np.loadtxt(IDS,dtype=int).tolist();pred={'K1':[],'native16':[]}
    assert len(list((O/'records').glob('*.json')))==982
    for i in ids:
        rec=json.loads((O/'records'/f'{i}.json').read_text());path=O/'outputs'/rec['output_file']
        assert rec['image_id']==i and rec['K']==16 and rec['seed']==42 and sha(path)==rec['output_sha256']
        assert np.isfinite(rec['scores']).all() and rec['pick']==int(np.argmax(rec['scores'])) and not rec['GT_used_for_selection']
        with np.load(path) as f:
            assert hashes([f['k1']])[0]==rec['candidate_tensor_sha256'][0]
            assert hashes([f['selected']])[0]==rec['candidate_tensor_sha256'][rec['pick']]
            if i in plan['fixed_visual_ids']:assert hashes(f['all_candidates'])==rec['candidate_tensor_sha256']
            for key,field in [('K1','k1'),('native16','selected')]:pred[key].append(torch.from_numpy(f[field].copy()))
    cfg=json.loads((R/'reports/subject02-bf16-seed42/effective-config.json').read_text())
    with h5py.File(Path(cfg['imgs_dir'])/'nsd_stimuli.hdf5','r') as f:
        gt=torch.stack([torch.from_numpy(f['imgBrick'][i]).permute(2,0,1).float()/255 for i in ids])
    evaluator=m.MetricEvaluator(device='cuda');results={}
    for label in pred:
        images=m.resize_images(torch.stack(pred[label]),gt.shape[2:])
        with torch.no_grad():frame,summary=evaluator.compute_all_metrics(gt,images)
        frame.insert(0,'image_id',ids)
        for k,v in summary['mean'].to_dict().items():assert np.isfinite(frame[k]).all() and abs(frame[k].mean()-v)<1e-10
        frame.to_csv(O/f'{label}-per-image.csv',index=False);results[label]=summary['mean'].to_dict()
    write(O/'metrics.json',dict(n=982,negative_pool=982,feature_batch=16,metrics=results,
        delta={k:results['native16'][k]-results['K1'][k] for k in results['K1']},
        records_and_means_verified=True,visual_review_pending=True))


def controller():
    import fcntl
    lock=(O/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'started.json').exists()
    frozen()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    write(O/'started.json',dict(pid=os.getpid(),time=time.time()))
    for stage in ['preflight','generate','score']:
        children=[]
        for rank in ([0] if stage=='score' else [0,1]):
            command=[str(R/'runtime/env/bin/python'),'-B',str(Path(__file__)),stage,'--rank',str(rank)]
            log=(O/f'{stage}-{rank}.log').open('x')
            child=subprocess.Popen(command,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
            children.append((rank,child,log))
            write(O/f'{stage}-{rank}-started.json',dict(pid=child.pid,command=command,time=time.time()))
        while any(c.poll() is None for _,c,_ in children):
            write(O/'status.json',dict(phase=stage,children=[dict(rank=k,pid=c.pid,returncode=c.poll()) for k,c,_ in children],time=time.time()),True)
            time.sleep(15)
        for rank,c,log in children:
            log.close();write(O/f'{stage}-{rank}-exit.json',dict(returncode=c.returncode,time=time.time()))
        assert all(c.returncode==0 for _,c,_ in children),'Inspect failed stage before narrow recovery'
        if stage=='preflight':
            a=json.loads((O/'preflight-0.json').read_text());b=json.loads((O/'preflight-1.json').read_text())
            assert a['candidate_tensor_sha256']==b['candidate_tensor_sha256'] and a['pick']==b['pick']
            with np.load(O/'preflight-0.npz') as f:save_result(a['image_id'],f['candidates'],a['pick'],a['scores'],json.loads((O/'plan.json').read_text()))
    write(O/'status.json',dict(phase='metrics_complete_needs_visual_review',time=time.time()),True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','generate','score','controller']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    try:
        if a.mode in ['preflight','generate']:generate(a.rank,a.mode=='preflight')
        else:globals()[a.mode]()
    except Exception:
        import traceback
        write(O/f'{a.mode}-{a.rank}-failure-{int(time.time())}.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
