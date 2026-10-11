"""Fixed L450 best-of-eight inference; no training or ground-truth image selection."""
from pathlib import Path
import os, sys, json, hashlib, time, argparse, subprocess, shutil, fcntl, traceback
R = Path(__file__).resolve().parents[1]
B = R / 'benchmarks/subj01-v1'
O = R / 'reports/l450-best8-test982'
PY = R / 'runtime/env/bin/python'
OLD = R / 'reports/balanced-lr-from400/test982'
ENCROOT = R.parent / 'neuroadapter-subject1-research/repo/vendor/whole_brain_encoder'
BASEENC = B / 'imagenet-retrieval/local-fixed-v1/encoder.py'
IDS = B / 'test_full_ids.txt'
sys.path.insert(0, str(R))
os.environ.update(TORCH_HOME=str(R/'runtime/torch'), MPLCONFIGDIR=str(R/'runtime/matplotlib'),
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def digest(a): return hashlib.sha256(a.tobytes()).hexdigest()

def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(O.resolve())
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def npz(path, **data):
    import numpy as np
    path = Path(path)
    assert path.resolve().is_relative_to(O.resolve()) and not path.exists()
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def stamp(**kw): return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)

def diskcheck():
    free = shutil.disk_usage(R).free
    if free < 2 * 1024**3: raise RuntimeError(f'Disk reserve below 2 GiB: {free}; preserve all outputs')
    return free

def frozen():
    a=json.loads((O/'authorization.json').read_text())
    for path, expected in a['protected'].items(): assert sha(R/path)==expected, path
    assert sha(IDS)==a['ids_sha256']
    for path, expected in a['code'].items(): assert sha(R/path)==expected, path
    return a

def setup_torch(tf32):
    import torch
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=tf32
    torch.backends.cudnn.allow_tf32=tf32
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)

def seed_for(i, c):
    return int.from_bytes(hashlib.sha256(f'42:{i}:{c}'.encode()).digest()[:8], 'little') % (2**63-1)

def load_generation():
    import torch
    import decode_brain_adapter as d
    setup_torch(True)
    cfg=json.loads((R/'configs/stage1.json').read_text())
    argv=[]
    for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path']: argv += ['--'+k,str(cfg[k])]
    a=d.create_argument_parser().parse_args(argv)
    for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk',
              'condition_dim','num_decoder_queries','sub_approach']: setattr(a,k,cfg[k])
    a.split='test'; a.image_ids=str(IDS); a.num_workers=0; a.batch_size=1
    a.export_path=json.loads((O/'authorization.json').read_text())['weight']
    scheduler,tokenizer,text_encoder,vae,unet=d.load_diffusion_models(a.pretrained_model_name_or_path)
    image_proj,adapters=d.setup_ip_adapter_modules(a,unet,num_tokens=200)
    ds,loader,_=d.create_test_dataset(a,tokenizer,topk=100,batch_size=1)
    vertices=torch.load(Path(a.export_path).parent/'parcel_vertices.pt',weights_only=True)
    for h in ['lh','rh']:
        assert len(vertices[h])==100
        assert all(torch.equal(x,y) for x,y in zip(vertices[h],ds.get_selected_voxel_indices()[h]))
    adapter=d.BrainIPAdapter(unet,image_proj,adapters)
    gg=d.GuidanceGenerator(num_parcels=ds.num_parcels,max_voxels=ds.max_voxels,
        num_decoder_queries=a.num_decoder_queries,output_dim=a.condition_dim,sub_approach=a.sub_approach)
    weights=torch.load(a.export_path,map_location='cpu',weights_only=True)
    image_proj.load_state_dict(weights['image_proj'],strict=True)
    adapters.load_state_dict(weights['ip_adapter'],strict=True)
    gg.load_state_dict(weights['guidance_generator'],strict=True)
    del weights
    for m in [adapter,gg,vae,text_encoder]: m.to('cuda')
    d.freeze_models(adapter,gg,vae,text_encoder)
    models=dict(tokenizer=tokenizer,text_encoder=text_encoder,vae=vae,noise_scheduler=scheduler,
        brain_adapter=adapter,guidance_generator=gg,device=torch.device('cuda'),weight_dtype=torch.float32)
    return d,ds,loader,gg,models

def generate(preflight=False):
    import numpy as np, torch
    auth=frozen();diskcheck()
    d,ds,loader,gg,models=load_generation()
    checks=[]; begin=time.monotonic()
    with torch.no_grad():
        for pos,batch in enumerate(loader):
            i=int(batch['image_id'].item())
            if preflight and pos>=2: break
            if not preflight:
                while len(list((O/'queue').glob('*.npz'))) >= 4:
                    if (O/'STOP').exists(): raise RuntimeError('Controller requested stop')
                    time.sleep(2)
                if (O/'STOP').exists(): raise RuntimeError('Controller requested stop')
            diskcheck()
            brain=torch.cat([batch['brain_lh_f'],batch['brain_rh_f']],dim=1).to('cuda',dtype=torch.float32)
            embeds,_=gg(brain)
            with np.load(OLD/'decoded'/f'image_{i:06d}.npz') as z:
                original=z['candidate_images'][0].copy()
                assert int(z['seeds'][0])==seed_for(i,0) and int(z['image_id'])==i
            cands=[original];seeds=[seed_for(i,0)]
            if preflight:
                fresh=d.run_diffusion(embeds,torch.zeros((1,3,512,512),device='cuda'),models,
                    num_predictions=1,noise_factor=4,denoising_steps=50,
                    generator=torch.Generator(device='cpu').manual_seed(seeds[0]))[0]
                assert np.array_equal(fresh,original), 'K1 candidate0 replay mismatch'
                checks.append(dict(image_id=i,bitwise_equal=True,pixel_sha256=digest(fresh)))
            else:
                for c in range(1,8):
                    seeds.append(seed_for(i,c))
                    im=d.run_diffusion(embeds,torch.zeros((1,3,512,512),device='cuda'),models,
                        num_predictions=1,noise_factor=4,denoising_steps=50,
                        generator=torch.Generator(device='cpu').manual_seed(seeds[-1]))[0]
                    cands.append(im)
                assert len(set(seeds))==8 and len({digest(x) for x in cands})==8
                npz(O/'queue'/f'image_{i:06d}.npz',image_id=i,candidate_images=np.stack(cands),
                    seeds=np.asarray(seeds,dtype=np.int64),
                    brain_lh_f=batch['brain_lh_f'].numpy(),brain_rh_f=batch['brain_rh_f'].numpy())
                write(O/'generation-status.json',stamp(completed=pos+1,total=982,
                    candidates_per_sample=8,generated_new_candidates=7*(pos+1),
                    reused_k1_candidates=pos+1,elapsed=time.monotonic()-begin,free_bytes=diskcheck()))
                print('GENERATED',pos+1,i,flush=True)
            del brain,embeds
    if preflight:
        write(O/'generation-preflight.json',stamp(passed=True,checks=checks,
            max_gpu_bytes=torch.cuda.max_memory_allocated(),training_updates=0))
    else: write(O/'generation-complete.json',stamp(n=982,generated_new=6874,reused_k1=982))

def load_selector():
    import numpy as np,torch,types
    from torchvision import transforms
    from scipy.special import softmax
    setup_torch(False)
    vertices=torch.load(Path(json.loads((O/'authorization.json').read_text())['weight']).parent/'parcel_vertices.pt',weights_only=True)
    masks={}
    for h in ['lh','rh']:
        masks[h]=np.zeros(163842,dtype=bool)
        masks[h][torch.cat(vertices[h]).numpy()]=True
    mask_path=O/'selector-mask.npz'
    if not mask_path.exists(): npz(mask_path,**masks)
    else:
        with np.load(mask_path) as z:
            for h in masks: assert np.array_equal(z[h],masks[h])
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(425),transforms.CenterCrop(425),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def prep(images):
        out=[]
        for x in images:
            y=transform(x); z=torch.zeros(3,434,434);z[:,:425,:425]=y;out.append(z)
        return torch.stack(out)
    ids=np.loadtxt(IDS,dtype=int).reshape(-1)
    sample=np.stack([np.load(OLD/'decoded'/f'image_{i:06d}.npz')['candidate_images'][0] for i in ids[:8]])
    check=prep(sample)
    sys.path.insert(0,str(BASEENC.parent))
    source=BASEENC.read_text()
    patches={
      'for layer in [1, 3, 5, 7]:\n                for run in [1, 2]:':
      'for run in [1, 2]:\n                for layer in [1, 3, 5, 7]:',
      'self.members, self.audit = [], []':'self.members, self.audit, self.reference = [], [], []',
      'check_batch[:2].to(device)':'check_batch.to(device)',
      "self.audit.append(dict(hemi=hemi":"self.reference.append(expected.cpu().clone())\n                    self.audit.append(dict(hemi=hemi",
      "corr.append(np.load(folder / f'{hemi}_val_corr_nonavg.npy'))":
      "corr.append(np.nan_to_num(np.load(folder / f'{hemi}_val_corr_nonavg.npy')).astype(np.float64))",
      "self.weights = torch.softmax(20 * torch.from_numpy(np.stack(corr)), dim=0)[:, self.mask.cpu()].to(device)":
      "from scipy.special import softmax\n        self.weights = torch.from_numpy(softmax(20*np.stack(corr),axis=0))[:,self.mask.cpu()]",
      "result = (torch.stack(parts) * self.weights[:, None, :]).sum(dim=0)":
      "result = (torch.stack(parts,dim=1).cpu() * self.weights[None,:,:]).sum(dim=1)",
    }
    for old,new in patches.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    module=types.ModuleType('best8_masked_encoder');module.__file__=str(BASEENC)
    exec(compile(source,str(BASEENC),'exec'),module.__dict__)
    module.MASK=mask_path
    halves=[module.HalfEncoder(h,'cuda:0',check) for h in ['lh','rh']]
    # Author transform/padding equality, without instantiating/downloading wrapper.
    sys.path.insert(0,str(ENCROOT))
    from datasets.nsd import nsd_dataset_custom
    native_ds=nsd_dataset_custom(sample,transform=transform)
    native=torch.stack([native_ds[j][0] for j in range(len(sample))])
    assert torch.equal(native,check)
    errors=[]
    for half in halves:
        pred=half.predict(check)
        reference=(torch.stack(half.reference,dim=1)*half.weights[None,:,:]).sum(dim=1).numpy()
        assert np.allclose(pred,reference,atol=2e-4,rtol=2e-4)
        errors.append(float(np.max(np.abs(pred-reference))))
    import decode_brain_adapter as d
    # Compare selected candidate using full author forward vs optimized forward
    # on all eight fixed candidate images for eight measured test fMRI queries.
    from types import SimpleNamespace
    cfg=json.loads((R/'configs/stage1.json').read_text())
    ds=d.nsd_topk_parcel_dataset(SimpleNamespace(subj=1,hemi=None,backbone_arch='dinov2_q',
       data_dir=cfg['data_dir'],imgs_dir=cfg['imgs_dir'],parcel_dir=cfg['parcel_dir'],
       tokenizer=None,gen_size=512),split='test',topk=100)
    mapids={int(v):j for j,v in enumerate(ds.base_dataset.split_presented_imgs)}
    def full(parts):
        result={}
        for h,arr in zip(['lh','rh'],parts):
            z=torch.zeros(len(arr),163842,dtype=torch.float64)
            z[:,torch.from_numpy(masks[h])]=torch.from_numpy(arr)
            result[h]=z
        return result
    predictions=full([h.predict(check) for h in halves])
    references=full([(torch.stack(h.reference,dim=1)*h.weights[None,:,:]).sum(dim=1).numpy() for h in halves])
    ranks=[]
    for i in ids[:8]:
        row=ds[mapids[int(i)]]
        batch={k:row[k][None] for k in ['brain_lh_f','brain_rh_f']}
        gt,p=d.process_brain_data_for_correlation(batch,ds,predictions,'cpu')
        _,ref=d.process_brain_data_for_correlation(batch,ds,references,'cpu')
        # Native function needs matching tensor dtypes for matmul; wrapper returns float64.
        # Preserve float32 ground-truth standardization then explicitly promote its result.
        scores=author_scores(gt,p); refs=author_scores(gt,ref)
        assert int(scores.argmax())==int(refs.argmax())
        ranks.append(dict(image_id=int(i),selected=int(scores.argmax()),
           full_forward_selected=int(refs.argmax()),max_correlation_error=float(np.max(np.abs(scores-refs)))))
    audit=stamp(passed=True,preprocessing_exact=True,
        source_sha256=sha(BASEENC),wrapper_sha256=sha(ENCROOT/'brain_encoder_wrapper.py'),
        patches=patches,member_audits=[a for h in halves for a in h.audit],
        ensemble_max_abs_errors=errors,rank_checks=ranks,mask_vertices={h:int(masks[h].sum()) for h in masks},
        ensemble_order='run1 layers1,3,5,7 then run2 layers1,3,5,7; scipy float64 softmax(20 val_corr)',
        memory_peak_bytes=torch.cuda.max_memory_allocated(),
        covariance='author compute_brain_correlation; explicit dtype promotion fixes float32/float64 matmul incompatibility',
        encoder_development_split_verified=False)
    dest=O/'selector-preflight.json'
    if not dest.exists():write(dest,audit)
    else:
        assert json.loads(dest.read_text())['passed']
    for h in halves:h.reference=[]
    return d,ds,prep,halves,full

def author_scores(gt,pred):
    import torch
    # Same operations/epsilon/std convention as decoder, with explicit dtype compatibility.
    g=(gt-gt.mean(dim=1,keepdim=True))/(gt.std(dim=1,keepdim=True)+1e-8)
    p=(pred-pred.mean(dim=1,keepdim=True))/(pred.std(dim=1,keepdim=True)+1e-8)
    return ((p@g.to(p.dtype).T).squeeze()/gt.size(1)).numpy()

def select(preflight=False):
    import numpy as np,torch
    frozen();diskcheck()
    d,ds,prep,halves,full=load_selector()
    if preflight:return
    ids=np.loadtxt(IDS,dtype=int).reshape(-1)
    fixed=set(map(int,ids[:32]))
    begin=time.monotonic()
    for pos,i in enumerate(ids):
        path=O/'queue'/f'image_{i:06d}.npz'
        while not path.exists():
            if (O/'STOP').exists():raise RuntimeError('Controller requested stop')
            time.sleep(2)
        diskcheck()
        with np.load(path) as z:
            assert int(z['image_id'])==i
            candidates=z['candidate_images'].copy();seeds=z['seeds'].copy()
            batch={k:torch.from_numpy(z[k].copy()) for k in ['brain_lh_f','brain_rh_f']}
        assert candidates.shape==(8,512,512,3) and candidates.dtype==np.uint8
        assert seeds.tolist()==[seed_for(int(i),c) for c in range(8)]
        x=prep(candidates)
        predictions=full([half.predict(x) for half in halves])
        gt,p=d.process_brain_data_for_correlation(batch,ds,predictions,'cpu')
        scores=author_scores(gt,p);assert np.isfinite(scores).all()
        idx=int(np.argmax(scores))
        candidate_hashes=[digest(c) for c in candidates]
        target=O/'selected'/path.name
        npz(target,image_id=int(i),candidate_images=candidates[idx:idx+1],seeds=seeds[idx:idx+1],
            selected_index=idx,brain_correlations=scores,all_seeds=seeds)
        # Preserve all candidates for the predeclared first32 image IDs only.
        if int(i) in fixed:
            npz(O/'fixed32'/path.name,image_id=int(i),candidate_images=candidates,seeds=seeds,
                selected_index=idx,brain_correlations=scores)
        record=dict(image_id=int(i),selected_index=idx,seeds=seeds.tolist(),
            brain_correlations=scores.tolist(),candidate_pixel_sha256=candidate_hashes,
            selected_pixel_sha256=digest(candidates[idx]),output_sha256=sha(target),
            measured_brain_sha256=digest(gt.numpy()),no_gt_image_in_selection=True)
        assert record['selected_pixel_sha256']==candidate_hashes[idx]
        write(O/'records'/f'{i}.json',record)
        # Delete only this task's consumed queue spool, after durable result+audit.
        assert path.parent.resolve()==(O/'queue').resolve()
        path.unlink()
        write(O/'selection-status.json',stamp(completed=pos+1,total=982,
            elapsed=time.monotonic()-begin,free_bytes=diskcheck()))
        print('SELECTED',pos+1,int(i),idx,float(scores[idx]),flush=True)
    write(O/'selection-complete.json',stamp(n=982,K=8,selection='max predicted/measured fMRI correlation'))

def score():
    import numpy as np
    frozen();diskcheck()
    ids=np.loadtxt(IDS,dtype=int).reshape(-1)
    for i in ids:
        record=json.loads((O/'records'/f'{i}.json').read_text())
        path=O/'selected'/f'image_{i:06d}.npz'
        assert sha(path)==record['output_sha256']
        assert record['selected_index']==int(np.argmax(record['brain_correlations']))
    source=(B/'evaluate_k1.py').read_text()
    replacements={
      "B/'outputs/neuroadapter'":"Path("+repr(str(O/'selected'))+")",
      "out=B/'metrics'/a.method/(a.output_tag or a.ids.stem)":"out=Path("+repr(str(O/'metrics'))+")",
      "'K':1,'selection':'none'":"'K':8,'selection':'author brain-encoder Pearson argmax; one selected image per ID'",
    }
    for old,new in replacements.items():
        assert source.count(old)==1,old;source=source.replace(old,new)
    write(O/'evaluator-adaptation.json',dict(source_sha256=sha(B/'evaluate_k1.py'),patches=replacements,
       metric_formula_unchanged=True,negative_pool=982))
    target=O/'evaluate-isolated.py'
    assert not target.exists()
    target.write_text("exec(compile("+repr(source)+","+repr(str(B/'evaluate_k1.py'))+",'exec'),"
        "{'__name__':'__main__','__file__':"+repr(str(B/'evaluate_k1.py'))+"})")
    with (O/'score.log').open('x') as f:
        rc=subprocess.run([str(PY),'-B',str(target),'--method','neuroadapter','--ids',str(IDS),
            '--feature-batch','16'],cwd=R,stdout=f,stderr=subprocess.STDOUT).returncode
    write(O/'score-exit.json',stamp(returncode=rc));assert rc==0
    frozen()
    old=json.loads((OLD/'metrics/metrics.json').read_text())
    new=json.loads((O/'metrics/metrics.json').read_text())
    write(O/'comparison.json',stamp(K1=old['mean'],best8=new['mean'],
        delta={k:new['mean'][k]-old['mean'][k] for k in old['mean']},
        same_weight=True,candidate0_matches_K1=True,
        training_updates=0,atlas_review_pending=True,
        limitations=['Previously viewed test set; not independent blind testing',
            'Official encoder checkpoint development split unknown',
            'Fixed per-ID seeds and serial candidate generation; not authors exact historical random stream',
            'One subject, one training seed; no significance claim']))

def controller():
    frozen()
    lock=(O/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'controller-started.json').exists()
    write(O/'controller-started.json',stamp(pid=os.getpid(),command=sys.argv))
    try:
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            write(O/'controller-status.json',stamp(status='waiting_for_idle_gpus'))
            time.sleep(60)
        diskcheck()
        assert json.loads((O/'generation-preflight.json').read_text())['passed']
        assert json.loads((O/'selector-preflight.json').read_text())['passed']
        children=[]
        for mode,gpu in [('select','1'),('generate','0')]:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            log=(O/f'{mode}.log').open('x')
            cmd=[str(PY),'-B',str(Path(__file__).resolve()),mode]
            p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((mode,p,log))
            write(O/f'{mode}-started.json',stamp(pid=p.pid,command=cmd,gpu=gpu))
        write(O/'controller-status.json',stamp(status='generating_and_selecting'))
        seen=set()
        while True:
            failed=False
            for mode,p,log in children:
                rc=p.poll()
                if rc is not None and mode not in seen:
                    write(O/f'{mode}-exit.json',stamp(returncode=rc));seen.add(mode)
                if rc not in (None,0):failed=True
            if failed:
                (O/'STOP').write_text('A child failed; inspect logs before explicit recovery.')
                for mode,p,log in children:
                    if p.poll() is None:p.wait(timeout=240)
                raise RuntimeError('Pipeline child failed; existing outputs preserved')
            if len(seen)==2:break
            time.sleep(10)
        for _,_,f in children:f.close()
        frozen()
        write(O/'controller-status.json',stamp(status='scoring'))
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
        with (O/'score-stage.log').open('x') as f:
            rc=subprocess.run([str(PY),'-B',str(Path(__file__).resolve()),'score'],
                cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        assert rc==0,rc
        write(O/'controller-status.json',stamp(status='metrics_complete_review_pending',exit_code=0))
    except BaseException:
        write(O/'controller-status.json',stamp(status='failed',traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['generation-preflight','selector-preflight','generate','select','score','controller'])
    mode=parser.parse_args().mode
    try:
        if mode=='generation-preflight':generate(True)
        elif mode=='selector-preflight':select(True)
        else:globals()[mode]()
    except BaseException:
        traceback.print_exc()
        raise
