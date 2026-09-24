"""Fixed-position content interchange around the unchanged author decoder."""
import argparse, hashlib, json, sys, random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
REPORT=ROOT/'reports/hl-content-interchange-l120'
WORK=RUN/'comparison/hl-content-interchange-l120'
def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--content-condition',choices=['AA','BB','BA','AB'],required=True)
    p.add_argument('--study-config',help='Explicit bounded replication config; absent preserves original pilot semantics')
    opt,rest=p.parse_known_args()
    import decode_brain_adapter as d
    args=d.create_argument_parser().parse_args(rest)
    assert args.no_selection and args.split=='val' and args.batch_size==1 and args.num_workers==0
    assert args.num_predictions==2 and args.denoising_steps==50 and args.noise_factor==4 and args.seed==42
    assert args.saved_epochs=='120' and args.precision=='fp32'
    assert Path(args.export_path)==RUN/'exports/epoch-120/inference.pt'
    out=Path(args.decoded_stimuli_dir).resolve()
    ids=np.loadtxt(args.image_ids,dtype=int).reshape(-1).tolist()
    if opt.study_config:
        cfg_path=Path(opt.study_config).resolve()
        study_report=ROOT/'reports/hl-content-replication-l120'
        study_work=RUN/'comparison/hl-content-replication-l120'
        assert cfg_path.is_relative_to(study_report)
        config=json.loads(cfg_path.read_text())
        assert config['study']=='L120-HL-content-replication' and config['mode'] in ['check','formal']
        check=config['mode']=='check'
        assert opt.content_condition in (['AA','BB'] if check else ['BA','AB','BB'])
        assert config['allowed_conditions']==(['AA','BB'] if check else ['BA','AB','BB'])
        assert Path(args.image_ids).resolve()==Path(config['target_ids']).resolve()
        assert ids==np.loadtxt(config['target_ids'],dtype=int).reshape(-1).tolist()
        assert out==Path(config['output_root'])/opt.content_condition and out.is_relative_to(study_work)
        assert not out.exists()
        mapping_path=Path(config['mapping_path'])
        assert hashlib.sha256(mapping_path.read_bytes()).hexdigest()==config['mapping_sha256']
        assert hashlib.sha256((ROOT/'reports/ibbi-roi-l120-pilot/roi_mapping.json').read_bytes()).hexdigest()==config['roi_mapping_sha256']
        assert config['model_sha256']=='bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798'
        mapping={r['target_image_id']:r['brain_source_image_id'] for r in json.loads(mapping_path.read_text())}
        full=set(np.loadtxt(ROOT/'configs/val_ids.txt',dtype=int).tolist())
        assert set(ids)<=full and len(ids)==len(set(ids)) and all(i!=mapping[i] for i in ids)
        if check:
            assert ids==np.loadtxt(ROOT/'configs/decode2_ids.txt',dtype=int).tolist()
            assert mapping=={r['target_image_id']:r['brain_source_image_id'] for r in json.loads((ROOT/'reports/brain-correspondence-l120/brain-map.json').read_text())}
        else:
            assert len(ids)==len(mapping)==len(set(mapping.values()))==100
            assert set(mapping)==set(ids) and not set(ids)&set(mapping.values()) and set(mapping.values())<=full
    else:
        assert out.is_relative_to(WORK) and not out.exists()
        check=opt.content_condition in ['AA','BB']
        assert ids==np.loadtxt(ROOT/'configs'/('decode2_ids.txt' if check else 'monitor100_ids.txt'),dtype=int).reshape(-1).tolist()
        mapping={r['target_image_id']:r['brain_source_image_id'] for r in json.loads((ROOT/'reports/brain-correspondence-l120/brain-map.json').read_text())}
    hl=json.loads((ROOT/'reports/ibbi-roi-l120-pilot/roi_mapping.json').read_text())['groups']['HL']
    assert len(hl)==78 and sum(i<100 for i in hl)==37
    u=[i for i in range(200) if i not in hl];assert len(u)==122 and 22 in hl and 143 in hl
    out.mkdir(parents=True)
    state={};original_create=d.create_test_dataset;original_diffuse=d.run_diffusion
    def create(*a,**kw):
        dataset,loader,indices=original_create(*a,**kw)
        lookup={int(v):i for i,v in enumerate(dataset.base_dataset.split_presented_imgs)}
        selected=list(loader.dataset.indices)
        assert [int(dataset.base_dataset.split_presented_imgs[i]) for i in selected]==ids
        class Paired(torch.utils.data.Dataset):
            def __len__(self):return len(selected)
            def __getitem__(self,index):
                target=dataset[selected[index]];i=int(target['image_id']);b=mapping[i];assert i!=b
                # Isolate any dataset helper randomness from the original target stream.
                cpu=torch.get_rng_state();npstate=np.random.get_state();pystate=random.getstate()
                try:donor=dataset[lookup[b]]
                finally:
                    torch.set_rng_state(cpu);np.random.set_state(npstate);random.setstate(pystate)
                assert int(donor['image_id'])==b
                state.clear();state.update(image_id=i,donor_id=b,candidate=0,
                    donor_input=torch.cat([donor['brain_lh_f'],donor['brain_rh_f']],dim=0).unsqueeze(0),
                    target_input_sha=sha(torch.cat([target['brain_lh_f'],target['brain_rh_f']],dim=0).unsqueeze(0)))
                return target
        return dataset,torch.utils.data.DataLoader(Paired(),batch_size=1,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available()),indices
    def diffuse(ea,img,models,**kw):
        assert ea.shape==(1,200,768) and not torch.is_grad_enabled() and torch.count_nonzero(img)==0
        assert all(not m.training and all(not q.requires_grad for q in m.parameters()) for m in models.values() if isinstance(m,torch.nn.Module))
        i=state['image_id'];k=state['candidate'];assert k in (0,1)
        seed=int.from_bytes(hashlib.sha256(f'42:{i}:{k}'.encode()).digest()[:8],'little')%(2**63-1)
        generator=kw['generator'];assert generator.initial_seed()==seed
        gs=generator.get_state().clone();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        if k==0:
            donor=state['donor_input'].to(models['device'],dtype=models['weight_dtype'])
            eb,_=models['guidance_generator'](donor)
            assert eb.shape==ea.shape and torch.isfinite(eb).all()
            state['eb']=eb;state['donor_input_sha']=sha(donor)
        eb=state['eb']
        assert torch.equal(gs,generator.get_state()) and torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        ah,bh=sha(ea),sha(eb);condition=opt.content_condition
        e=(ea if condition[1]=='A' else eb).clone()
        e[:,hl,:]=(ea if condition[0]=='A' else eb)[:,hl,:]
        assert torch.equal(e[:,hl,:],(ea if condition[0]=='A' else eb)[:,hl,:])
        assert torch.equal(e[:,u,:],(ea if condition[1]=='A' else eb)[:,u,:])
        assert sha(ea)==ah and sha(eb)==bh
        calls=[0]
        def projection(module,inputs):
            if calls[0]%2==0:assert torch.equal(inputs[0],e)
            else:assert torch.count_nonzero(inputs[0])==0
            calls[0]+=1
        hook=models['brain_adapter'].image_proj_model.register_forward_pre_hook(projection)
        try:result=original_diffuse(e,img,models,**kw)
        finally:hook.remove()
        assert calls[0]==100 and sha(ea)==ah and sha(eb)==bh
        exact=None
        if check:
            ref=(RUN/'decoded/epoch-120/val' if condition=='AA' else RUN/'comparison/brain-correspondence-l120/permuted500')/f'image_{i:06d}.npz'
            with np.load(ref) as z:
                assert int(z['seeds'][k])==seed
                assert np.array_equal(result[0],z['candidate_images'][k]),('Pixel reproduction failed',condition,i,k)
            exact=True
        row=dict(image_id=i,donor_id=state['donor_id'],candidate=k,condition=condition,seed=seed,
            E_A_sha256=ah,E_B_sha256=bh,E_mix_sha256=sha(e),donor_brain_sha256=state['donor_input_sha'],
            hl_same_parcel_source_exact=True,u_same_parcel_source_exact=True,source_E_unmodified=True,
            extra_encoding_rng_unchanged=True,generator_before_diffusion_sha256=sha(gs),
            projection_calls_verified=calls[0],unconditional_E_zero=True,pixel_exact=exact)
        with (out/'content-audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        state['candidate']+=1
        return result
    d.create_test_dataset=create;d.run_diffusion=diffuse
    sys.argv=[str(ROOT/'decode_brain_adapter.py')]+rest
    d.main()
    audits=[json.loads(x) for x in (out/'content-audit.jsonl').read_text().splitlines()]
    assert len(audits)==len(ids)*2
    (out/'content-protocol.json').write_text(json.dumps(dict(condition=opt.content_condition,images=ids,
        hl_tokens=hl,complement_tokens=u,position='GuidanceGenerator E output before original ImageProjModel',
        generated_candidates=len(audits),study_config=opt.study_config,mode='check' if check else 'formal'),indent=2)+'\n')
if __name__=='__main__':main()
