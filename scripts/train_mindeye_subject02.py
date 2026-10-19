"""Independent MindEye1 high/low training using pinned author loss blocks."""
import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import types

from mindeye_subject02_common import R,O,D,S,COMPAT,sha,write,source_components,loss_function,index_tar,sample
import numpy as np
import torch
import torch.nn.functional as F


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def rng_restore(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])


def fingerprint(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(str((y.shape,y.dtype)).encode());h.update(y.numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in sorted(x.items(),key=lambda p:str(p[0])):h.update(str(k).encode());visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def build(kind):
    models,utils,augs,patches=source_components()
    seed=42 if kind=='high' else 0;epochs=240 if kind=='high' else 120;batch=32 if kind=='high' else 8
    utils.seed_everything(seed,cudnn_deterministic=False)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    if kind=='high':torch.use_deterministic_algorithms(True)
    frozen={}
    if kind=='high':
        voxel=models.BrainNetwork(in_dim=14278,out_dim=257*768,clip_size=768,use_projector=True)
        net=models.VersatileDiffusionPriorNetwork(dim=768,depth=6,dim_head=64,heads=12,causal=False,num_tokens=257,learned_query_mode='pos_emb')
        model=models.BrainDiffusionPrior(net=net,image_embed_dim=768,condition_on_text_encodings=False,
            timesteps=100,cond_drop_prob=.2,image_embed_scale=None,voxel2clip=voxel).cuda()
        from torch.utils.checkpoint import checkpoint
        model._mindeye_checkpointing_enabled=True
        def checkpoint_module(module):
            original=module.forward
            def forward(*args,**kwargs):
                if model._mindeye_checkpointing_enabled and torch.is_grad_enabled():
                    return checkpoint(original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
                return original(*args,**kwargs)
            module.forward=forward
        for layer in model.net.causal_transformer.layers:
            for module in layer:checkpoint_module(module)
        checkpoint_module(model.voxel2clip.projector)
        frozen['clip']=models.Clipper('ViT-L/14',device=torch.device('cuda'),hidden_state=True,norm_embs=True)
        named=[list(model.net.named_parameters()),list(model.voxel2clip.named_parameters())]
        max_lr=3e-4
    else:
        from diffusers import AutoencoderKL
        vae=AutoencoderKL(down_block_types=['DownEncoderBlock2D']*4,up_block_types=['UpDecoderBlock2D']*4,
            block_out_channels=[128,256,512,512],layers_per_block=2,sample_size=256)
        vae.load_state_dict(torch.load(D/'models/sd_image_var_autoenc.pth',map_location='cpu',weights_only=True),strict=True)
        vae.requires_grad_(False).eval().cuda();frozen['vae']=vae
        module=types.ModuleType('mindeye_subject02_convnext');module.__file__=str(S/'convnext.py')
        exec(compile((S/'convnext.py').read_text(),module.__file__,'exec'),module.__dict__)
        def strict_backbone_load(self,path):
            state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model']
            state={k.replace('module.',''):v for k,v in state.items()}
            result=self.load_state_dict(state,strict=False)
            assert not result.missing_keys,result.missing_keys
            assert all(k.startswith(('projector.','classifier.')) for k in result.unexpected_keys),result.unexpected_keys
            self.loaded_key_audit=dict(missing=result.missing_keys,unused_author_global_projector=result.unexpected_keys)
        module.ConvnextXL.init_weights=strict_backbone_load
        cnx=module.ConvnextXL(D/'models/convnext_xlarge_alpha0.75_fullckpt.pth')
        cnx.requires_grad_(False).eval().cuda();frozen['cnx']=cnx
        model=models.Voxel2StableDiffusionModel(in_dim=14278,use_cont=True,ups_mode='4x').cuda()
        named=[list(model.named_parameters())];max_lr=5e-4
    for frozen_model in frozen.values():frozen_model.requires_grad_(False).eval()
    no_decay=['bias','LayerNorm.bias','LayerNorm.weight'];groups=[]
    for values in named:
        groups.extend([{'params':[p for n,p in values if not any(x in n for x in no_decay)],'weight_decay':1e-2},
                       {'params':[p for n,p in values if any(x in n for x in no_decay)],'weight_decay':0.}])
    optimizer=torch.optim.AdamW(groups,lr=max_lr if kind=='high' else 1e-3,foreach=False,fused=kind=='high')
    steps=8559//batch
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=max_lr,total_steps=epochs*steps,
        final_div_factor=1000,last_epoch=-1,pct_start=2/epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=kind=='high')
    compute,loss_sha=loss_function(kind,model,frozen,utils,augs[kind].cuda(),epochs)
    config=dict(kind=kind,subject=2,seed=seed,epochs=epochs,batch_size=batch,world_size=1,
        train_images=8559,val_images=300,train_steps_per_epoch=steps,max_lr=max_lr,
        precision='fp16_autocast_with_fp32_parameters' if kind=='high' else 'fp32',
        weight_decay=.01,optimizer='AdamW; original groups; fused=True for high, foreach=False for low; same hyperparameters',
        gradient_clipping=False,loss_block_sha256=loss_sha,source_patches=patches,
        activation_checkpointing='Prior attention/feedforward and voxel projector; nonreentrant with RNG preservation' if kind=='high' else False,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        source_files={name:sha(S/name) for name in ['models.py','utils.py','Train_MindEye.py','train_autoencoder.py','convnext.py']},
        compatibility='Pinned pure-Python author dependencies including diffusers0.13.0 with runtime torch2.11; not bit-exact historical execution',
        data_order='Original resampled-shard and500-buffer shuffle; deterministic shard RNG based on seed/epoch; num_workers0',
        validation='All300 held-out images; full300 high-level contrastive pool via chunked features; fixed RNG restored afterwards',
        primary_endpoint='fixed last epoch; best held-out diagnostic inference state retained separately; no test-based switch')
    if kind=='high':config['deterministic_algorithms']=True
    return model,frozen,utils,augs[kind],optimizer,scheduler,scaler,compute,config


def train_batches(config,epoch):
    import webdataset as wds
    paths=[str(D/f'webdataset_avg_split/train/train_subj02_{i}.tar') for i in range(18)]
    dataset=wds.WebDataset(paths,resampled=True,nodesplitter=lambda urls:urls)
    sampler=dataset.pipeline[0]
    assert isinstance(sampler,wds.ResampledShards)
    sampler.deterministic=True;sampler.worker_seed=lambda:config['seed'];sampler.epoch=epoch-1
    dataset=dataset.shuffle(500,initial=500,rng=random.Random(42)).decode('torch').rename(
        images='jpg;png',voxels='nsdgeneral.npy',coco='coco73k.npy').to_tuple('voxels','images','coco')
    dataset=dataset.batched(config['batch_size'],partial=True).with_epoch(config['train_steps_per_epoch'])
    return torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=0)


def group_probes(model,kind):
    groups=['net.','voxel2clip.lin0.','voxel2clip.lin1.','voxel2clip.projector.'] if kind=='high' else ['lin0.','upsampler.','maps_projector.']
    return {g:hashlib.sha256(b''.join(p.detach().reshape(-1)[:64].cpu().numpy().tobytes() for n,p in model.named_parameters() if n.startswith(g))).hexdigest() for g in groups}


def step(model,frozen,optimizer,scheduler,scaler,compute,config,batch,epoch,index,check_groups=False):
    model.train();voxel,image,ids=batch;voxel=voxel.cuda();image=image.cuda()
    optimizer.zero_grad(set_to_none=True)
    before=group_probes(model,config['kind']) if check_groups else None
    with torch.autocast('cuda',dtype=torch.float16,enabled=config['kind']=='high'):
        loss,parts=compute(voxel,image,epoch,index)
    assert torch.isfinite(loss).all(),'Nonfinite training loss'
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    norms=[torch.linalg.vector_norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]
    norm=torch.linalg.vector_norm(torch.stack(norms));finite=bool(torch.isfinite(norm))
    scale_before=scaler.get_scale()
    if config['kind']=='low':assert finite,'Nonfinite FP32 gradient'
    scaler.step(optimizer);scaler.update();skipped=scaler.get_scale()<scale_before
    assert skipped or finite,'Nonfinite gradient update was not skipped'
    if not skipped:scheduler.step()
    assert all(p.grad is None for f in frozen.values() for p in f.parameters())
    changed=None
    if check_groups and not skipped:
        after=group_probes(model,config['kind']);changed={g:before[g]!=after[g] for g in before}
        assert all(changed.values()),changed
    value=dict(loss=float(loss.detach()),components={k:float(v) for k,v in parts.items()},
        gradient_norm=float(norm) if finite else None,gradients_finite=finite,skipped=skipped,
        scale_before=scale_before,scale_after=scaler.get_scale(),lr=optimizer.param_groups[0]['lr'],
        group_updates=changed,batch_n=len(image))
    del loss,parts,voxel,image
    return value


def snapshot(model,optimizer,scheduler,scaler,config,epoch,updates,skips,best):
    return dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        lr_scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),
        epoch=epoch,updates=updates,skips=skips,best_validation=best,config=config,
        loader={'next_epoch':epoch+1,'seed':config['seed'],'deterministic_resampled_shards':True})


def restore(state,model,optimizer,scheduler,scaler):
    model.load_state_dict(state['model_state_dict'],strict=True)
    optimizer.state.clear();gc.collect();torch.cuda.empty_cache()
    optimizer.load_state_dict(state['optimizer_state_dict']);scheduler.load_state_dict(state['lr_scheduler'])
    scaler.load_state_dict(state['scaler']);rng_restore(state['rng'])


def validate(model,frozen,utils,augment,config,rows,epoch):
    saved=rng_state();utils.seed_everything(config['seed']+7000,cudnn_deterministic=False);model.eval()
    predictions=[];targets=[];total=0.;n=0
    try:
        with torch.no_grad():
            for start in range(0,len(rows),16):
                data=[sample(r) for r in rows[start:start+16]]
                voxel=torch.stack([x[0] for x in data]).cuda().mean(1).float()
                image=torch.stack([x[1] for x in data]).cuda()
                if config['kind']=='high':
                    with torch.autocast('cuda',dtype=torch.float16):
                        image=augment(image);target=frozen['clip'].embed_image(image).float()
                        v,p=model.voxel2clip(voxel);v=v.view(len(voxel),257,768)
                        prior_loss,_=model(text_embed=v,image_embed=target)
                        predictions.append(F.normalize(p.flatten(1),dim=-1).float().cpu())
                        targets.append(F.normalize(target.flatten(1),dim=-1).float().cpu())
                    total+=float(prior_loss)*len(voxel)
                else:
                    image=F.interpolate(image,(512,512),mode='bilinear',align_corners=False,antialias=True)
                    target=frozen['vae'].encode(2*image-1).latent_dist.mode()*.18215
                    pred=model(voxel);total+=float(F.mse_loss(pred,target))*len(voxel)
                n+=len(voxel)
                del voxel,image,target
        assert n==300
        if config['kind']=='high':
            pred=torch.cat(predictions).cuda();target=torch.cat(targets).cuda()
            if epoch<int(.33*config['epochs']):nce=utils.mixco_nce(pred,target,temp=.006)
            else:
                temps=utils.cosine_anneal(.004,.0075,config['epochs']-int(.33*config['epochs']))
                nce=utils.soft_clip_loss(pred,target,temp=float(temps[epoch-int(.33*config['epochs'])]))
            result={'loss':float(nce)+30*total/n,'nce':float(nce),'prior':total/n,
                'cosine':float(F.cosine_similarity(pred,target).mean()),'n':n,'contrastive_pool':n}
        else:result={'loss':total/n,'latent_mse':total/n,'n':n}
        assert all(np.isfinite(v) for v in result.values())
        return result
    finally:rng_restore(saved);model.train()


def preflight(kind):
    target=O/f'preflight-{kind}';target.mkdir(exist_ok=False)
    model,frozen,utils,aug,opt,sched,scaler,compute,config=build(kind)
    rows=index_tar(D/'webdataset_avg_split/train/train_subj02_17.tar')
    data=[sample(x) for x in rows[:config['batch_size']]]
    batch=(torch.stack([x[0] for x in data]),torch.stack([x[1] for x in data]),torch.tensor([x[2] for x in data]))
    checkpoint_parity=None
    if kind=='high':
        saved_rng=rng_state();model.train();gradients=None;losses=[];differences=[]
        for enabled in [False,True]:
            model._mindeye_checkpointing_enabled=enabled;rng_restore(saved_rng);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss,_=compute(batch[0][:4].cuda(),batch[1][:4].cuda(),0,0)
            loss.backward();losses.append(float(loss.detach()))
            if gradients is None:
                gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            else:
                for n,p in model.named_parameters():
                    if n in gradients:
                        assert p.grad is not None,n
                        current=p.grad.detach().cpu();reference=gradients[n]
                        assert torch.isfinite(current).all() and torch.isfinite(reference).all(),n
                        if not torch.equal(current,reference):
                            delta=current-reference
                            differences.append({'name':n,'max_abs':float(delta.abs().max()),
                                'relative_L2':float(delta.norm()/(reference.norm()+1e-20)),
                                'reference_max_abs':float(reference.abs().max())})
                    else:assert p.grad is None,n
            del loss
        write(target/'activation-checkpoint-diagnostic.json',{'losses':losses,'gradient_differences':differences})
        assert all(x['name']=='net.causal_transformer.rel_pos_bias.relative_attention_bias.weight'
                   and x['max_abs']<=1e-6 and x['relative_L2']<=1e-5 for x in differences),differences
        assert losses[0]==losses[1]
        checkpoint_parity={'same_loss':True,'all_parameter_gradients_exact':not differences,
            'numerical_gradient_agreement':True,'shared_embedding_gradient_tolerance':{'max_abs':1e-6,'relative_L2':1e-5},
            'gradient_differences':differences,'probe_batch':4}
        del gradients;opt.zero_grad(set_to_none=True);rng_restore(saved_rng);gc.collect();torch.cuda.empty_cache()
    records=[];successful=False
    for i in range(8):
        row=step(model,frozen,opt,sched,scaler,compute,config,batch,0,i,True);records.append(row)
        if not row['skipped']:successful=True;break
    assert successful,'AMP failed to produce a finite update in preflight'
    opt.zero_grad(set_to_none=True)
    saved=target/'roundtrip.pth'
    torch.save(snapshot(model,opt,sched,scaler,config,0,1,sum(r['skipped'] for r in records),None),saved)
    saved_sha=sha(saved)
    continuation=step(model,frozen,opt,sched,scaler,compute,config,batch,0,i+1,True)
    assert not continuation['skipped']
    expected_model=fingerprint(model.state_dict());expected_opt=fingerprint(opt.state_dict())
    expected_rng=fingerprint(rng_state());expected_scaler=scaler.state_dict();expected_sched=sched.state_dict()
    state=torch.load(saved,map_location='cpu',weights_only=False,mmap=True)
    restore(state,model,opt,sched,scaler);del state
    repeated=step(model,frozen,opt,sched,scaler,compute,config,batch,0,i+1,True)
    assert not repeated['skipped'] and repeated['loss']==continuation['loss']
    assert fingerprint(model.state_dict())==expected_model
    assert fingerprint(opt.state_dict())==expected_opt
    assert fingerprint(rng_state())==expected_rng and scaler.state_dict()==expected_scaler and sched.state_dict()==expected_sched
    write(target/'audit.json',dict(passed=True,config=config,actual_GPU=torch.cuda.get_device_name(),
        real_batch_shape=list(batch[0].shape),source_shard_sha256=sha(D/'webdataset_avg_split/train/train_subj02_17.tar'),
        initial_steps=records,continuous_step=continuation,restored_step=repeated,
        activation_checkpoint_parity=checkpoint_parity,
        resumed_model_optimizer_RNG_scaler_scheduler_exact=True,frozen_no_gradients=True,
        checkpoint_sha256=saved_sha,checkpoint_bytes=saved.stat().st_size,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),time=time.time()))
    # This isolated scratch state has served its verified roundtrip; keep its digest/audit.
    assert saved.resolve().parent==target.resolve() and saved.name=='roundtrip.pth' and not saved.is_symlink()
    saved.unlink()
    print('PREFLIGHT_PASSED',kind,flush=True)


def train(kind):
    import fcntl
    import math
    report=O/kind;report.mkdir(exist_ok=True)
    lock=(report/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (report/'started.json').exists(),'No blind restart or duplicate launch'
    ready=json.loads((O/'data-ready.json').read_text());assert ready['complete']
    locked=json.loads((O/'frozen-plan.json').read_text())
    assert locked['independent_validation_authorized'] and locked['test_in_training']==False
    pre=json.loads((O/f'preflight-{kind}/audit.json').read_text());assert pre['passed']
    assert sha(O/'data-index.json')==ready['data_index_sha256']
    data=json.loads((O/'data-index.json').read_text());allowed={x['image_id'] for x in data['train']}
    output=R/f'runs/mindeye-subject02-{kind}';output.mkdir(exist_ok=False)
    model,frozen,utils,aug,opt,sched,scaler,compute,config=build(kind)
    assert config==pre['config']
    write(report/'started.json',dict(pid=os.getpid(),config=config,output=str(output),time=time.time()))
    updates=skips=0;best=float('inf');consecutive_skips=0
    for epoch in range(config['epochs']):
        for filename,digest in locked['code_sha256'].items():assert sha(R/filename)==digest,filename
        for item in locked['asset_stats']:
            stat=(R/item['path']).stat()
            assert stat.st_size==item['bytes'] and stat.st_mtime_ns==item['mtime_ns'],item['path']
        started=time.time();stats=[];id_hash=hashlib.sha256()
        for i,batch in enumerate(train_batches(config,epoch)):
            ids=batch[2].reshape(-1).tolist();assert set(ids)<=allowed
            id_hash.update(np.asarray(ids,dtype=np.int64).tobytes())
            row=step(model,frozen,opt,sched,scaler,compute,config,batch,epoch,i,updates==0)
            row.update(epoch=epoch+1,step=i+1,time=time.time())
            skips+=int(row['skipped']);updates+=int(not row['skipped'])
            consecutive_skips=consecutive_skips+1 if row['skipped'] else 0
            assert consecutive_skips<=4,'Persistent AMP overflow; preserve and inspect'
            with (report/'updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            stats.append(row)
            if i%10==0:
                write(report/'status.json',dict(phase='training',epoch=epoch+1,target=config['epochs'],
                    step=i+1,steps_per_epoch=config['train_steps_per_epoch'],updates=updates,skips=skips,
                    loss=row['loss'],lr=row['lr'],pid=os.getpid(),time=time.time()),True)
                print('UPDATE',kind,epoch+1,i+1,row['loss'],updates,skips,flush=True)
        assert len(stats)==config['train_steps_per_epoch']
        opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        val=validate(model,frozen,utils,aug,config,data['val'],epoch)
        improved=val['loss']<best
        if improved:
            best=val['loss'];temp=output/'best-inference.pth.tmp'
            torch.save(dict(model_state_dict=model.state_dict(),epoch=epoch+1,validation=val),temp)
            temp.replace(output/'best-inference.pth')
        checkpoint_bytes=pre['checkpoint_bytes']
        assert shutil.disk_usage(R).free>checkpoint_bytes+25*1024**3,'Checkpoint disk reserve reached'
        target=output/f'checkpoint-{epoch+1:03d}.pth';temp=target.with_name(target.name+'.tmp')
        state=snapshot(model,opt,sched,scaler,config,epoch,updates,skips,best);torch.save(state,temp)
        saved_model=fingerprint(state['model_state_dict']);saved_optimizer=fingerprint(state['optimizer_state_dict'])
        loaded=torch.load(temp,map_location='cpu',weights_only=False,mmap=True)
        assert fingerprint(loaded['model_state_dict'])==saved_model and fingerprint(loaded['optimizer_state_dict'])==saved_optimizer
        assert fingerprint(loaded['rng'])==fingerprint(state['rng']) and loaded['scaler']==state['scaler'] and loaded['loader']==state['loader']
        del loaded,state;temp.replace(target)
        complete=dict(epoch=epoch+1,updates=updates,skips=skips,checkpoint=target.name,sha256=sha(target),
            model_sha256=saved_model,optimizer_sha256=saved_optimizer,train_loss=float(np.mean([x['loss'] for x in stats])),
            validation=val,train_id_order_sha256=id_hash.hexdigest(),elapsed_seconds=time.time()-started,time=time.time())
        write(output/f'checkpoint-{epoch+1:03d}.COMPLETE.json',complete)
        with (report/'epochs.jsonl').open('a') as f:f.write(json.dumps(complete)+'\n')
        # Only this new run's oldest rolling continuation file is retired after verified save.
        old_epoch=epoch-1
        if old_epoch>=1:
            old=output/f'checkpoint-{old_epoch:03d}.pth'
            assert old.resolve().parent==output.resolve() and not old.is_symlink()
            if old.exists():old.unlink()
        write(report/'status.json',dict(phase='epoch_complete',**complete,pid=os.getpid(),target=config['epochs']),True)
        if (report/'STOP_AFTER_EPOCH').exists():
            write(report/'intentional-stop.json',complete);return
    final=output/'last.pth';torch.save(dict(model_state_dict=model.state_dict(),epoch=config['epochs'],config=config),final)
    write(report/'training-complete.json',dict(complete=True,epochs=config['epochs'],updates=updates,skips=skips,
        final_sha256=sha(final),final_weight=str(final.relative_to(R)),test_used=False,time=time.time()))
    print('TRAINING_COMPLETE',kind,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','train']);p.add_argument('--kind',choices=['high','low'],required=True);a=p.parse_args()
    try:globals()[a.mode](a.kind)
    except Exception:
        target=O/f'{a.mode}-{a.kind}-failure-{int(time.time())}.json'
        write(target,dict(traceback=traceback.format_exc(),time=time.time(),pid=os.getpid()))
        raise
