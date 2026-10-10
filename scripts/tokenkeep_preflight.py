"""Zero-update engineering checks using only fixed training examples."""
import os,sys,json,time,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,numpy as np
import train_brain_adapter as t
import tokenkeep_objective as obj
from branch_lr_resume import equal

def main():
    report=ROOT/'reports/tokenkeep-from450'
    assert not (report/'preflight.json').exists()
    start=time.monotonic()
    torch.manual_seed(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    args=t.create_argument_parser().parse_args([item for k,v in json.loads((ROOT/'configs/stage1.json').read_text()).items() for item in ['--'+k,str(v)]])
    scheduler,tokenizer,text,vae,unet=t.load_pretrained_models(args)
    proj,modules,_=t.setup_ip_adapter(unet,args)
    dargs=SimpleNamespace(subj=1,backbone_arch='dinov2_q',data_dir=args.data_dir,imgs_dir=args.imgs_dir,parcel_dir=args.parcel_dir,hemi=None,tokenizer=tokenizer,gen_size=512,num_decoder_queries=50)
    ds=t.nsd_topk_parcel_dataset(dargs,split='train',transform=None,topk=100)
    assert len(ds)==8500
    ids=list(map(int,ds.base_dataset.split_presented_imgs))
    chosen=sorted(ids,key=lambda i:hashlib.sha256(f'spatial-preflight/{i}'.encode()).hexdigest())[:4]
    samples=[ds[ids.index(i)] for i in chosen]
    batch={k:torch.stack([s[k] if torch.is_tensor(s[k]) else torch.as_tensor(s[k]) for s in samples]) for k in ['img_ipadapter','text_input_ids','brain_lh_f','brain_rh_f']}
    guidance=t.GuidanceGenerator(num_parcels=ds.num_parcels,max_voxels=ds.max_voxels,num_decoder_queries=50,output_dim=768,sub_approach='linear_projection')
    model=t.NeuroAdapter(unet,proj,modules)
    cp=ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-from400/checkpoint-450'
    model.load_state_dict(torch.load(cp/'pytorch_model.bin',map_location='cpu',weights_only=False),strict=True)
    guidance.load_state_dict(torch.load(cp/'pytorch_model_1.bin',map_location='cpu',weights_only=False),strict=True)
    ref=torch.load(ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-from400/exports/epoch-450/parcel_vertices.pt',map_location='cpu',weights_only=False)
    equal(ref,ds.get_selected_voxel_indices())
    for m in [model,guidance,vae,text]:m.to('cuda',dtype=torch.float32)
    vae.requires_grad_(False);text.requires_grad_(False);vae.eval();text.eval()
    groups={'mapper':list(guidance.parameters()),'projection':list(proj.parameters()),'ip_kv':list(modules.parameters())}
    params=[p for v in groups.values() for p in v if p.requires_grad]
    assert all(not p.requires_grad for n,p in unet.named_parameters() if '.processor.' not in n)
    acc=SimpleNamespace(device=torch.device('cuda'),print=lambda *a,**k:None)
    state=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def reset():
        torch.set_rng_state(state[0]);torch.cuda.set_rng_state_all(state[1])
        for p in params:p.grad=None
    def call():return t.process_training_batch(batch,vae,scheduler,text,guidance,model,torch.float32,acc,ds)
    reset();x=call();base_value=float(x);x.backward()
    base_rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    grad_sample=[p.grad.detach().clone() for p in [groups['mapper'][0],groups['projection'][0],groups['ip_kv'][0]]]
    del x
    t.apply_fmri_token_dropout=lambda tokens:obj.apply(tokens,keep_min=0)
    reset();x=call();assert float(x)==base_value;x.backward()
    equal(base_rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    for a,p in zip(grad_sample,[groups['mapper'][0],groups['projection'][0],groups['ip_kv'][0]]):equal(a,p.grad)
    del x,grad_sample
    t.apply_fmri_token_dropout=obj.apply
    reset();torch.cuda.reset_peak_memory_stats();x=call();new_value=float(x);assert torch.isfinite(x);x.backward()
    equal(base_rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    norms={n:sum(float(p.grad.square().sum()) for p in group if p.grad is not None)**.5 for n,group in groups.items()}
    assert all(np.isfinite(v) and v>0 for v in norms.values())
    assert all(p.grad is None for m in [vae,text] for p in m.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert torch.cuda.max_memory_allocated()<22*2**30, 'Preflight memory margin inadequate'
    result=dict(passed=True,new_optimizer_updates=0,train_only_ids=chosen,keep_min_zero_exact_loss_rng_gradients=True,keep_min_half_same_rng_consumption=True,strict_source_loaded=True,parcel_identity=True,frozen_parameters_no_grad=True,base_loss=base_value,new_loss=new_value,group_grad_norms=norms,max_allocated_bytes=torch.cuda.max_memory_allocated(),keep_min=obj.KEEP_MIN,seconds=time.monotonic()-start)
    (report/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
