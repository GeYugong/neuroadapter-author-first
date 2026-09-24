"""Read-only author attention capture and fixed E-token masks around the original decoder."""
import argparse, ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/ibbi-roi-l120-pilot'
RUN=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
WORK=RUN/'comparison/ibbi-roi-l120-pilot'
STEPS=[1,10,20,30,40,50]
def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def author_functions():
    # Only these pure original functions: avoid the plotting entry's hardcoded cwd/pycortex/dataset.
    tree=ast.parse((ROOT/'plot_attn_dynamics.py').read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
              ['integrate_attention_layers','create_roi_attention_map']]
    assert len(selected)==2
    ns={'np':np,'torch':torch}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'author_attention_functions','exec'),ns)
    return ns
class Recorder:
    def __init__(self,adapter,decoder,groups,record):
        self.record=record;self.adapter=adapter;self.groups=groups;self.functions=author_functions()
        self.layers=[n for n,m in adapter.unet.named_modules() if isinstance(getattr(m,'processor',None),decoder.IPAttnProcessor)]
        assert len(self.layers)>0
        self.spatial={};self.layer_info={};self.active=False;self.max_error=0.
        adapter.register_forward_pre_hook(self.before_step)
        adapter.register_forward_hook(self.after_step)
        adapter.image_proj_model.register_forward_hook(self.projection)
        adapter.unet.register_forward_pre_hook(self.unet_input)
        for name,module in adapter.unet.named_modules():
            if module.__class__.__name__=='Transformer2DModel':
                module.register_forward_pre_hook(self.grid_hook(name))
            if name in self.layers:module.register_forward_hook(self.attention_hook(name))
    def begin(self,e):
        assert not self.active
        self.active=True;self.expected=e;self.timesteps=[];self.bt=[];self.layer_summary=[]
        self.selected_ram=[];self.ram_total=np.zeros((len(self.groups),512,512),dtype=np.float64)
        self.unavailable=[];self.max_error=0.;self.rng_checks=0;self.condition_checks=0
    def before_step(self,module,inputs):
        assert self.active and len(inputs)==4
        assert not torch.is_grad_enabled() and not module.training
        assert torch.equal(inputs[3],self.expected)
        self.timesteps.append(int(inputs[1]));self.projections=[];self.seen=[]
        if self.record:
            self.bsum=np.zeros(200,dtype=np.float64);self.qsum=0
            self.ram=np.zeros((len(self.groups),512,512),dtype=np.float64)
            self.layer_step={}
    def projection(self,module,inputs,output):
        assert self.active and len(self.projections)<2 and not module.training
        if not self.projections:assert torch.equal(inputs[0],self.expected)
        else:assert torch.count_nonzero(inputs[0])==0
        self.projections.append(output.detach())
    def unet_input(self,module,inputs):
        assert len(self.projections)==2
        combined=inputs[2]
        assert combined.shape[0]==2 and combined.shape[-2]>200
        # Prove actual CFG ordering from the actual two projection calls, not shape alone.
        assert torch.equal(combined[0,-200:],self.projections[1][0])
        assert torch.equal(combined[1,-200:],self.projections[0][0])
        assert torch.equal(combined[0,:-200],combined[1,:-200])
        self.condition_checks+=1
        self.projections.clear()
    def grid_hook(self,name):
        def hook(module,inputs):
            x=inputs[0];assert x.ndim==4 and x.shape[0]==2
            self.spatial[name]=tuple(x.shape[-2:])
        return hook
    def attention_hook(self,name):
        def hook(module,inputs,output):
            a=module.processor.ip_attn_map
            assert a is not None and a.ndim==4 and a.shape[0]==2 and a.shape[-1]==200
            assert name not in self.seen
            self.seen.append(name)
            if self.record:
                cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
                a=a[1].detach() # actual conditional branch established by unet_input above
                assert torch.isfinite(a).all() and a.min()>=0
                error=float((a.sum(-1)-1).abs().max())
                assert error<2e-5,(name,error)
                self.max_error=max(self.max_error,error)
                parent=max((n for n in self.spatial if name.startswith(n+'.')),key=len)
                h,w=self.spatial[parent];heads,q,p=a.shape
                assert h*w==q and h==w
                info={'heads':heads,'queries':q,'height':h,'width':w,'parcels':p}
                if name in self.layer_info:assert self.layer_info[name]==info
                self.layer_info[name]=info
                b=self.functions['integrate_attention_layers']({name:a}).cpu().numpy()
                self.layer_step[name]=b
                self.bsum+=q*b;self.qsum+=q
                for gi,(g,tokens) in enumerate(self.groups.items()):
                    if not tokens or float(a[:,:,tokens].sum())<=0:
                        self.unavailable.append({'step':len(self.timesteps),'layer':name,'group':g})
                        self.ram[gi]=np.nan;continue
                    ram=self.functions['create_roi_attention_map']({name:a},tokens,(512,512))
                    assert np.isfinite(ram).all() and abs(float(ram.sum())-1)<2e-5
                    self.ram[gi]+=ram
                assert torch.equal(cpu_rng,torch.get_rng_state())
                assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
                self.rng_checks+=1
            # Same cleanup used by the author's extract hook; no module output is modified.
            module.processor.ip_attn_map=None
            return None
        return hook
    def after_step(self,module,inputs,output):
        assert set(self.seen)==set(self.layers) and self.condition_checks==len(self.timesteps)
        if self.record:
            b=self.bsum/self.qsum
            assert abs(b.sum()-1)<2e-5
            self.bt.append(b.astype(np.float32))
            self.layer_summary.append(np.stack([self.layer_step[n] for n in self.layers]))
            ram=self.ram/len(self.layers)
            masses=ram.sum(axis=(1,2),keepdims=True)
            ram=np.divide(ram,masses,out=np.full_like(ram,np.nan),where=masses>0)
            self.ram_total+=ram
            if len(self.timesteps) in STEPS:self.selected_ram.append(ram.astype(np.float32))
        return None
    def finish(self,path):
        assert len(self.timesteps)==50 and self.condition_checks==50
        if self.record:
            assert len(self.selected_ram)==6
            np.savez_compressed(path,brain_attention=np.stack(self.bt),
                layer_brain_attention=np.stack(self.layer_summary),
                roi_spatial_maps=np.stack(self.selected_ram+[self.ram_total.astype(np.float32)/50]),
                saved_step_indices=np.array(STEPS+[-1]),actual_timesteps=np.array(self.timesteps),
                group_names=np.array(list(self.groups)),layer_names=np.array(self.layers))
            save(path.with_suffix('.json'),{'layers':self.layer_info,'max_parcel_mass_error':self.max_error,
                'rng_readonly_checks':self.rng_checks,'conditional_branch_checks':self.condition_checks,
                'unavailable_spatial_maps':self.unavailable,'all_heads_equal':len({v['heads'] for v in self.layer_info.values()})==1,
                'B_definition':'Mean over heads within layer, then query-count weighted layers.',
                'RAM_definition':'Author layer ROI/head mean, actual square grid, bilinear512, per-layer L1, equal layers; then mean over all50 steps.',
                'step_minus_one':'Mean of already normalized per-step RAMs; quantitative arrays have no display minmax.'})
        self.active=False;self.expected=None
def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot-condition',required=True);p.add_argument('--record-attention',action='store_true')
    extra,remaining=p.parse_known_args()
    import decode_brain_adapter as decoder
    args=decoder.create_argument_parser().parse_args(remaining)
    assert args.no_selection and args.split=='val' and args.batch_size==1 and args.num_workers==0
    assert args.num_predictions==2 and args.denoising_steps==50 and args.noise_factor==4 and args.seed==42
    assert args.saved_epochs=='120' and args.precision=='fp32'
    assert Path(args.export_path)==RUN/'exports/epoch-120/inference.pt'
    mapping=json.loads((REPORT/'roi_mapping.json').read_text())
    masks=mapping['mask_sets'];assert extra.pilot_condition=='BASE' or extra.pilot_condition in masks
    assert extra.record_attention==(extra.pilot_condition=='BASE')
    mask=masks.get(extra.pilot_condition,[])
    groups={g:mapping['groups'][g] for g in ['V1','V2','V3','V4','Face','Body','Scene','Word']}
    destination=Path(args.decoded_stimuli_dir).resolve()
    assert destination.is_relative_to(WORK) and not destination.exists()
    requested=np.loadtxt(args.image_ids,dtype=int).reshape(-1).tolist()
    allowed=np.loadtxt(ROOT/'configs'/('atlas_val32_ids.txt' if extra.record_attention else 'monitor100_ids.txt'),dtype=int).tolist()
    assert set(requested)<=set(allowed) and len(requested)==len(set(requested))
    destination.mkdir(parents=True)
    if extra.record_attention:(destination/'attention').mkdir()
    state={'id':None,'candidate':0};original_create=decoder.create_test_dataset
    original_diffusion=decoder.run_diffusion
    def create_dataset(*a,**kw):
        dataset,loader,indices=original_create(*a,**kw)
        selected=list(loader.dataset.indices)
        assert [int(dataset.base_dataset.split_presented_imgs[i]) for i in selected]==requested
        class Tracked(torch.utils.data.Dataset):
            def __len__(self):return len(selected)
            def __getitem__(self,index):
                target=dataset[selected[index]]
                state['id']=int(target['image_id']);state['candidate']=0
                return target
        replacement=torch.utils.data.DataLoader(Tracked(),batch_size=1,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
        return dataset,replacement,indices
    recorders=[]
    def register(adapter):
        r=Recorder(adapter,decoder,groups,extra.record_attention);recorders.append(r)
    def diffuse(brain_embeds,img_ip,models_dict,**kw):
        assert len(recorders)==1 and brain_embeds.shape==(1,200,768)
        assert torch.count_nonzero(img_ip)==0 and not torch.is_grad_enabled()
        assert all(not m.training and all(not q.requires_grad for q in m.parameters())
                   for m in models_dict.values() if isinstance(m,torch.nn.Module))
        before=tsha(brain_embeds)
        e=brain_embeds.clone()
        e[:,mask,:]=0
        keep=[i for i in range(200) if i not in mask]
        assert torch.equal(e[:,keep,:],brain_embeds[:,keep,:])
        assert torch.count_nonzero(e[:,mask,:])==0 and before==tsha(brain_embeds)
        assert torch.equal(e,brain_embeds) if not mask else True
        image_id=state['id'];k=state['candidate'];assert image_id is not None and k in (0,1)
        expected_seed=int.from_bytes(hashlib.sha256(f'42:{image_id}:{k}'.encode()).digest()[:8],'little')%(2**63-1)
        assert kw['generator'].initial_seed()==expected_seed
        r=recorders[0];r.begin(e)
        result=original_diffusion(e,img_ip,models_dict,**kw)
        r.finish(destination/'attention'/f'image_{image_id:06d}-candidate-{k}.npz')
        if extra.record_attention:
            with np.load(RUN/f'decoded/epoch-120/val/image_{image_id:06d}.npz') as base:
                assert np.array_equal(result[0],base['candidate_images'][k]),('Recording changed generation',image_id,k)
        row={'image_id':image_id,'candidate':k,'seed':expected_seed,'condition':extra.pilot_condition,
             'mask_tokens':mask,'E_original_sha256':before,'E_intervened_sha256':tsha(e),
             'unselected_rows_exact':True,'selected_rows_zero':True,'conditional_checks':r.condition_checks,
             'unconditional_E_original_zeros':True,'baseline_pixel_exact':True if extra.record_attention else None}
        with (destination/'input-audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        state['candidate']+=1
        return result
    decoder.create_test_dataset=create_dataset;decoder.run_diffusion=diffuse
    decoder.BrainIPAdapter._register_block_hook=register
    sys.argv=[str(ROOT/'decode_brain_adapter.py')]+remaining
    decoder.main()
    save(destination/'pilot-protocol.json',{'condition':extra.pilot_condition,'mask':mask,'attention':extra.record_attention,
        'mapping_sha256':hashlib.sha256((REPORT/'roi_mapping.json').read_bytes()).hexdigest(),
        'original_decoder_sha256':hashlib.sha256((ROOT/'decode_brain_adapter.py').read_bytes()).hexdigest(),
        'images':requested,'candidates':len(requested)*2,'mask_position':'E after GuidanceGenerator and before original ImageProjModel; original unconditional branch unchanged'})
if __name__=='__main__':main()

