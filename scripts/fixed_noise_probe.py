"""Read-only paired denoising probes; original models/data, explicit random streams."""
import os,sys,json,hashlib,time,argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import train_brain_adapter as t

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def seed_for(image,seed,kind):
    return int.from_bytes(hashlib.sha256(f'balanced-v1/{image}/{seed}/{kind}'.encode()).digest()[:8],'little')%(2**63-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--budget-seconds',type=int,default=14400);a=p.parse_args()
    cp=Path(a.checkpoint).resolve();out=Path(a.output).resolve()
    assert cp.is_relative_to(ROOT/'runs') and (cp/'COMPLETE').exists()
    assert out.is_relative_to(ROOT/'reports/balanced-lr-from400')
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'started.json').exists(),'Existing attempt must be inspected, not overwritten'
    started=time.monotonic();save(out/'started.json',dict(checkpoint=str(cp),pid=os.getpid(),wall_time=time.time()))
    torch.manual_seed(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    c=json.loads((ROOT/'configs/stage1.json').read_text())
    cli=[]
    for k,v in c.items():cli+=['--'+k,str(v)]
    args=t.create_argument_parser().parse_args(cli)
    scheduler,tokenizer,text,vae,unet=t.load_pretrained_models(args)
    assert scheduler.config.num_train_timesteps==1000 and scheduler.config.prediction_type=='epsilon'
    proj,modules,_=t.setup_ip_adapter(unet,args)
    dargs=SimpleNamespace(subj=1,backbone_arch='dinov2_q',data_dir=args.data_dir,imgs_dir=args.imgs_dir,parcel_dir=args.parcel_dir,hemi=None,tokenizer=tokenizer,gen_size=512,num_decoder_queries=50)
    datasets={s:t.nsd_topk_parcel_dataset(dargs,split=s,transform=None,topk=100) for s in ['train','val']}
    assert len(datasets['train'])==8500 and len(datasets['val'])==500
    assert set(map(int,datasets['val'].base_dataset.split_presented_imgs))==set(np.loadtxt(ROOT/'configs/val_ids.txt',dtype=int))
    manifest={}
    for split,ds in datasets.items():
        ids=list(map(int,ds.base_dataset.split_presented_imgs))
        selected=sorted(ids,key=lambda i:hashlib.sha256(f'balanced-probe-v1/{split}/{i}'.encode()).hexdigest())[:128]
        manifest[split]={'ids':selected,'donor_ids':selected[1:]+selected[:1]}
    assert not set(manifest['train']['ids'])&set(manifest['val']['ids'])
    encoded=json.dumps(manifest,sort_keys=True).encode();save(out/'query-manifest.json',manifest)
    guidance=t.GuidanceGenerator(num_parcels=datasets['train'].num_parcels,max_voxels=datasets['train'].max_voxels,num_decoder_queries=50,output_dim=768,sub_approach='linear_projection')
    model=t.NeuroAdapter(unet,proj,modules)
    model.load_state_dict(torch.load(cp/'pytorch_model.bin',map_location='cpu',weights_only=False),strict=True)
    guidance.load_state_dict(torch.load(cp/'pytorch_model_1.bin',map_location='cpu',weights_only=False),strict=True)
    reference=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100/exports/epoch-400/parcel_vertices.pt'
    from branch_lr_resume import equal
    vertices=torch.load(reference,map_location='cpu',weights_only=False)
    for ds in datasets.values():equal(vertices,ds.get_selected_voxel_indices())
    for module in [model,guidance,vae,text]:module.requires_grad_(False);module.eval();module.to('cuda',dtype=torch.float32)
    counts={};sums={};forward_count=0;budget_hit=False
    save(out/'load-audit.json',dict(strict_complete_model=True,strict_guidance=True,parcel_identity=True,query_manifest_sha256=hashlib.sha256(encoded).hexdigest(),checkpoint_model_sha256=sha(cp/'pytorch_model.bin'),guidance_sha256=sha(cp/'pytorch_model_1.bin'),new_training_updates=0,timesteps=[50,250,500,900],seeds=[42,43],backend='FP32 original TF32 enabled; deterministic algorithms'))
    with torch.inference_mode(),(out/'per-probe.jsonl').open('x') as log:
        for split,ds in datasets.items():
            mapping={int(i):j for j,i in enumerate(ds.base_dataset.split_presented_imgs)}
            for image,donor in zip(manifest[split]['ids'],manifest[split]['donor_ids']):
                if time.monotonic()-started>a.budget_seconds:budget_hit=True;break
                sample=ds[mapping[image]];wrong=ds[mapping[donor]]
                assert int(sample['image_id'])==image and int(wrong['image_id'])==donor and image!=donor
                brain=torch.cat([sample['brain_lh_f'],sample['brain_rh_f']],dim=0).unsqueeze(0).cuda().float()
                wrongbrain=torch.cat([wrong['brain_lh_f'],wrong['brain_rh_f']],dim=0).unsqueeze(0).cuda().float()
                img=sample['img_ipadapter'].unsqueeze(0).cuda().float()
                assert all(torch.isfinite(x).all() for x in [brain,wrongbrain,img])
                true_tokens=guidance(brain)[0];wrong_tokens=guidance(wrongbrain)[0]
                embedding=text(sample['text_input_ids'].reshape(1,-1).cuda())[0]
                dist=vae.encode(img).latent_dist
                for seed in [42,43]:
                    generator=torch.Generator(device='cuda').manual_seed(seed_for(image,seed,'latent'))
                    latent=dist.sample(generator=generator)*vae.config.scaling_factor
                    gen=torch.Generator().manual_seed(seed_for(image,seed,'mask'))
                    ratio=torch.rand(1,1,generator=gen)
                    mask=(torch.rand(1,200,generator=gen)<=ratio).cuda().unsqueeze(-1)
                    variants={'true_full':true_tokens,'wrong_full':wrong_tokens,'zero':torch.zeros_like(true_tokens),'true_dropout':true_tokens*mask,'wrong_dropout':wrong_tokens*mask}
                    for timestep in [50,250,500,900]:
                        generator=torch.Generator(device='cuda').manual_seed(seed_for(image,seed,f'noise-{timestep}'))
                        noise=torch.randn(latent.shape,generator=generator,device='cuda',dtype=torch.float32)
                        ts=torch.tensor([timestep],device='cuda');noisy=scheduler.add_noise(latent,noise,ts)
                        predictions={name:model(noisy,ts,embedding,tokens)[0] for name,tokens in variants.items()}
                        for name,pred in predictions.items():
                            assert torch.isfinite(pred).all()
                            mse=(pred-noise).square().mean().item();weighted=mse*t.min_snr_loss_weights(ts,scheduler,gamma=5.).item()
                            row=dict(split=split,image_id=image,donor_id=donor,seed=seed,timestep=timestep,condition=name,mse=mse,weighted_mse=weighted,kept_tokens=int(mask.sum()),token_rms=variants[name].square().mean().sqrt().item(),projected_token_rms=proj(variants[name]).square().mean().sqrt().item(),prediction_minus_zero_rms=(pred-predictions['zero']).square().mean().sqrt().item())
                            log.write(json.dumps(row)+'\n');key=f'{split}/{timestep}/{name}';counts[key]=counts.get(key,0)+1;sums[key]=sums.get(key,0)+mse;forward_count+=1
                        if forward_count>=32 and not (out/'throughput.json').exists():
                            torch.cuda.synchronize();elapsed=time.monotonic()-started
                            save(out/'throughput.json',dict(forward_count=forward_count,elapsed_including_load_seconds=elapsed,conservative_total_seconds=elapsed/forward_count*10240,max_allocated_bytes=torch.cuda.max_memory_allocated()))
                log.flush();save(out/'status.json',dict(forward_count=forward_count,last_split=split,last_id=image,elapsed_seconds=time.monotonic()-started))
            if budget_hit:break
    save(out/'summary.json',dict(complete=forward_count==10240,budget_hit=budget_hit,new_training_updates=0,forward_count=forward_count,expected_forward_count=10240,elapsed_seconds=time.monotonic()-started,mean_mse={k:sums[k]/counts[k] for k in sums},counts=counts,query_manifest_sha256=hashlib.sha256(encoded).hexdigest()))
    print('PROBE_FINISHED',forward_count,'budget_hit',budget_hit,flush=True)

if __name__=='__main__':main()
