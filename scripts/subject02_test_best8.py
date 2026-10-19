"""One final Subject2 test: fixed8 brain-response selection and shared candidate0 K1."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

R=Path(__file__).resolve().parents[1]
P=R/'reports/subject02-bf16-seed42'
O=P/'test982-best8'
PY=R/'runtime/env/bin/python'
IDS=R/'data/subject02/standard-test-ids.txt'
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts'))
os.environ.update(TORCH_HOME=str(R/'runtime/torch'),MPLCONFIGDIR=str(R/'runtime/matplotlib'),
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,value,replace=False):
    assert p.resolve().is_relative_to(O.resolve()) and (replace or not p.exists())
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)


def npz(p,**arrays):
    import numpy as np
    assert p.resolve().is_relative_to(O.resolve()) and not p.exists()
    temp=p.with_name(p.name+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    temp.replace(p)


def frozen():
    plan=json.loads((O/'plan.json').read_text())
    assert plan['selected_epoch'] in (109,300,372) and plan['n']==982 and plan['K']==8
    for name,known in plan['protected_sha256'].items():assert sha(R/name)==known,name
    assert json.loads((P/'final-selection.json').read_text())['confirmed']
    assert shutil.disk_usage(R).free>=25*1024**3
    return plan


def seed_for(i,c):
    return int.from_bytes(hashlib.sha256(f'42:{i}:{c}'.encode()).digest()[:8],'little')%(2**63-1)


def load_generation(plan):
    import torch
    import decode_brain_adapter as d
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    config=json.loads((P/'effective-config.json').read_text())
    argv=[]
    for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path']:argv+=['--'+k,str(config[k])]
    args=d.create_argument_parser().parse_args(argv)
    for k in ['subject_id','topk','condition_dim','num_decoder_queries','sub_approach']:setattr(args,k,config[k])
    args.split='test';args.image_ids=str(IDS);args.num_workers=0;args.batch_size=1
    weight=R/plan['weight']
    scheduler,tokenizer,text,vae,unet=d.load_diffusion_models(args.pretrained_model_name_or_path)
    projection,processors=d.setup_ip_adapter_modules(args,unet,num_tokens=200)
    ds,loader,_=d.create_test_dataset(args,tokenizer,topk=100,batch_size=1)
    saved=torch.load(weight.parent/'parcel_vertices.pt',weights_only=True)
    for h in ['lh','rh']:
        actual=ds.get_selected_voxel_indices()[h];assert len(saved[h])==len(actual)==100
        assert all(torch.equal(x,y) for x,y in zip(saved[h],actual))
    adapter=d.BrainIPAdapter(unet,projection,processors)
    guidance=d.GuidanceGenerator(num_parcels=200,max_voxels=ds.max_voxels,num_decoder_queries=args.num_decoder_queries,
                                 output_dim=args.condition_dim,sub_approach=args.sub_approach)
    weights=torch.load(weight,map_location='cpu',weights_only=True)
    projection.load_state_dict(weights['image_proj'],strict=True);processors.load_state_dict(weights['ip_adapter'],strict=True)
    guidance.load_state_dict(weights['guidance_generator'],strict=True)
    for model in [adapter,guidance,vae,text]:model.to('cuda')
    d.freeze_models(adapter,guidance,vae,text)
    models=dict(tokenizer=tokenizer,text_encoder=text,vae=vae,noise_scheduler=scheduler,
                brain_adapter=adapter,guidance_generator=guidance,device=torch.device('cuda'),weight_dtype=torch.float32)
    return d,ds,loader,guidance,models


def generate():
    import torch
    import numpy as np
    plan=frozen();d,ds,loader,guidance,models=load_generation(plan)
    start=time.time()
    with torch.no_grad():
        for position,batch in enumerate(loader):
            while len(list((O/'queue').glob('*.npz')))>=4:
                if (O/'STOP').exists():raise RuntimeError('Controller requested stop')
                time.sleep(2)
            if (O/'STOP').exists():raise RuntimeError('Controller requested stop')
            assert shutil.disk_usage(R).free>=25*1024**3
            i=int(batch['image_id'].item());path=O/'queue'/f'image_{i:06d}.npz'
            assert not (O/'records'/f'{i}.json').exists() and not path.exists()
            brain=torch.cat([batch['brain_lh_f'],batch['brain_rh_f']],dim=1).to('cuda',dtype=torch.float32)
            embeddings,_=guidance(brain);images=[];seeds=[]
            for candidate in range(8):
                seed=seed_for(i,candidate);seeds.append(seed)
                images.append(d.run_diffusion(embeddings,torch.zeros((1,3,512,512),device='cuda'),models,
                    num_predictions=1,noise_factor=4,denoising_steps=50,
                    generator=torch.Generator(device='cpu').manual_seed(seed))[0])
            assert len(set(seeds))==8 and len({hashlib.sha256(x.tobytes()).hexdigest() for x in images})==8
            npz(path,image_id=i,candidate_images=np.stack(images),seeds=np.asarray(seeds,dtype=np.int64),
                brain_lh_f=batch['brain_lh_f'].numpy(),brain_rh_f=batch['brain_rh_f'].numpy())
            write(O/'generation-status.json',dict(completed=position+1,n=982,K=8,elapsed=time.time()-start),True)
            print('GENERATED',position+1,i,flush=True)
    write(O/'generation-complete.json',dict(n=982,K=8,candidates=7856))


def select():
    import torch
    import numpy as np
    frozen()
    from subject02_selector import load_selector
    d,ds,prep,halves,full,correlate=load_selector()
    ids=np.loadtxt(IDS,dtype=int);start=time.time()
    # ds supplies only Subject2 parcel identities here; all measured test brain data come from the queue.
    for position,i in enumerate(ids):
        i=int(i);path=O/'queue'/f'image_{i:06d}.npz'
        while not path.exists():
            if (O/'STOP').exists():raise RuntimeError('Controller requested stop')
            time.sleep(2)
        with np.load(path) as payload:
            assert int(payload['image_id'])==i
            images=payload['candidate_images'].copy();seeds=payload['seeds'].copy()
            brain={k:torch.from_numpy(payload[k].copy()) for k in ['brain_lh_f','brain_rh_f']}
        assert images.shape==(8,512,512,3) and images.dtype==np.uint8
        assert seeds.tolist()==[seed_for(i,c) for c in range(8)]
        predictions=full([half.predict(prep(images)) for half in halves])
        gt,p=d.process_brain_data_for_correlation(brain,ds,predictions,'cpu')
        scores=correlate(gt,p);assert scores.shape==(8,) and np.isfinite(scores).all()
        winner=int(scores.argmax());pixels=[hashlib.sha256(x.tobytes()).hexdigest() for x in images]
        selected=O/'selected'/path.name;k1=O/'k1'/path.name
        npz(selected,image_id=i,candidate_images=images[winner:winner+1],seeds=seeds[winner:winner+1],
            selected_index=winner,all_seeds=seeds,brain_correlations=scores)
        npz(k1,image_id=i,candidate_images=images[:1],seeds=seeds[:1])
        archive=O/'candidates'/path.name
        assert path.parent.resolve()==(O/'queue').resolve() and not archive.exists()
        path.replace(archive)
        write(O/'records'/f'{i}.json',dict(image_id=i,selected_index=winner,seeds=seeds.tolist(),
              correlations=scores.tolist(),candidate_pixel_sha256=pixels,selected_npz_sha256=sha(selected),
              k1_npz_sha256=sha(k1),candidate_archive_sha256=sha(archive),
              measured_brain_sha256=hashlib.sha256(gt.numpy().tobytes()).hexdigest(),GT_image_used_for_selection=False))
        write(O/'selection-status.json',dict(completed=position+1,n=982,elapsed=time.time()-start),True)
        print('SELECTED',position+1,i,winner,flush=True)
    write(O/'selection-complete.json',dict(n=982,K=8,all_candidates_preserved=True))


def score():
    import numpy as np
    import pandas as pd
    import torch
    import h5py
    torch.set_num_threads(4);frozen()
    sys.path.insert(0,str(R/'benchmarks/subj01-v1'))
    import evaluate_stage  # installs the established frozen metric asset paths
    import metric_brain_adapter as metric
    from metric_feature_batches import install
    install(metric,16)
    ids=np.loadtxt(IDS,dtype=int);assert len(ids)==982
    all_seeds=set()
    for i in ids:
        rec=json.loads((O/'records'/f'{i}.json').read_text())
        assert rec['image_id']==i and rec['selected_index']==int(np.argmax(rec['correlations']))
        assert rec['seeds']==[seed_for(int(i),c) for c in range(8)] and not rec['GT_image_used_for_selection']
        assert not (all_seeds&set(rec['seeds']));all_seeds.update(rec['seeds'])
        for folder,key in [('selected','selected_npz_sha256'),('k1','k1_npz_sha256'),('candidates','candidate_archive_sha256')]:
            assert sha(O/folder/f'image_{i:06d}.npz')==rec[key]
        with np.load(O/'candidates'/f'image_{i:06d}.npz') as archive,np.load(O/'selected'/f'image_{i:06d}.npz') as chosen,np.load(O/'k1'/f'image_{i:06d}.npz') as first:
            assert int(archive['image_id'])==int(chosen['image_id'])==int(first['image_id'])==int(i)
            assert archive['seeds'].tolist()==rec['seeds'] and len(set(rec['candidate_pixel_sha256']))==8
            assert [hashlib.sha256(x.tobytes()).hexdigest() for x in archive['candidate_images']]==rec['candidate_pixel_sha256']
            winner=rec['selected_index']
            assert np.array_equal(chosen['candidate_images'][0],archive['candidate_images'][winner])
            assert np.array_equal(first['candidate_images'][0],archive['candidate_images'][0])
            assert chosen['seeds'].tolist()==[rec['seeds'][winner]] and first['seeds'].tolist()==rec['seeds'][:1]
            assert np.array_equal(chosen['brain_correlations'],np.asarray(rec['correlations']))
    assert len(all_seeds)==7856 and not list((O/'queue').glob('*.npz'))
    config=json.loads((P/'effective-config.json').read_text())
    with h5py.File(Path(config['imgs_dir'])/'nsd_stimuli.hdf5','r') as f:
        original=torch.stack([torch.from_numpy(f['imgBrick'][int(i)]).permute(2,0,1).float()/255 for i in ids])
    evaluator=metric.MetricEvaluator(device='cuda');results={}
    for label,folder in [('K1','k1'),('best8','selected')]:
        images=torch.stack([torch.from_numpy(np.load(O/folder/f'image_{i:06d}.npz')['candidate_images'][0].copy()) for i in ids]).permute(0,3,1,2).float()/255
        images=metric.resize_images(images,original.shape[2:])
        with torch.no_grad():frame,summary=evaluator.compute_all_metrics(original,images)
        frame.insert(0,'image_id',ids)
        for key,value in summary['mean'].to_dict().items():assert np.isfinite(frame[key]).all() and np.isclose(frame[key].mean(),value,atol=1e-10,rtol=0)
        frame.to_csv(O/f'{label}-per-image.csv',index=False)
        results[label]=summary['mean'].to_dict();del images
    write(O/'metrics.json',dict(n=982,negative_pool=982,feature_batch=16,means_verified=True,metrics=results,
          delta={k:results['best8'][k]-results['K1'][k] for k in results['K1']},K1_is_candidate0_of_best8=True,
          seed_count=len(all_seeds),test_used_for_weight_selection=False,visual_review_pending=True))


def controller():
    lock=(O/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'started.json').exists();frozen()
    for name in ['queue','selected','k1','records','candidates']:(O/name).mkdir(exist_ok=False)
    write(O/'started.json',dict(pid=os.getpid(),time=time.time()))
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
    children=[]
    for mode,gpu in [('generate',0),('select',1)]:
        command=[str(PY),'-B',str(Path(__file__).resolve()),mode]
        log=(O/f'{mode}.log').open('x')
        child=subprocess.Popen(command,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(O/f'{mode}-started.json',dict(command=command,pid=child.pid,gpu=gpu,time=time.time()));children.append((mode,child,log))
    while any(p.poll() is None for _,p,_ in children):
        if any(p.poll() not in (None,0) for _,p,_ in children):(O/'STOP').touch(exist_ok=True)
        write(O/'status.json',dict(phase='generation_and_selection',time=time.time(),pid=os.getpid(),children=[dict(mode=n,pid=p.pid,returncode=p.poll()) for n,p,_ in children]),True)
        time.sleep(15)
    for mode,child,log in children:
        log.close();write(O/f'{mode}-exit.json',dict(pid=child.pid,returncode=child.returncode,time=time.time()))
    assert all(p.returncode==0 for _,p,_ in children),'Inspect failed stage before any explicit recovery'
    frozen()
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
    with (O/'score.log').open('x') as log:
        child=subprocess.Popen([str(PY),'-B',str(Path(__file__).resolve()),'score'],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
        write(O/'score-started.json',dict(pid=child.pid,time=time.time()))
        write(O/'status.json',dict(phase='scoring',child_pid=child.pid,time=time.time()),True)
        code=child.wait()
    write(O/'score-exit.json',dict(returncode=code,time=time.time()));assert code==0
    frozen();write(O/'status.json',dict(phase='metrics_complete_needs_final_review',time=time.time(),pid=os.getpid()),True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['controller','generate','select','score']);args=p.parse_args()
    try:globals()[args.mode]()
    except Exception:
        target=O/f'{args.mode}-failure.json'
        if not target.exists():write(target,dict(time=time.time(),traceback=traceback.format_exc()))
        if args.mode=='controller':write(O/'status.json',dict(phase='failed_needs_inspection',time=time.time()),True)
        raise
