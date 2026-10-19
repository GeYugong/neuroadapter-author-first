"""Frozen validation500 K2 comparison after user-authorized early stopping."""
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

R = Path(__file__).resolve().parents[1]
O = R/'reports/subject02-bf16-seed42'
A = O/'validation-after-stop'
D = R/'runs/subject02-author-bf16-seed42'
PY = R/'runtime/env/bin/python'
IDS = R/'data/subject02/val_ids.txt'
EPOCHS = (109, 300, 372)
sys.path.insert(0, str(R))
os.environ.update(TORCH_HOME=str(R/'runtime/torch'), MPLCONFIGDIR=str(R/'runtime/matplotlib'),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(p, obj, replace=False):
    assert p.resolve().is_relative_to(A.resolve())
    assert replace or not p.exists(), f'Refusing overwrite: {p}'
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    temp.replace(p)


def status(phase, **kw):
    write(A/'status.json',dict(phase=phase,controller_pid=os.getpid(),time=time.time(),**kw),replace=True)


def frozen():
    plan=read(A/'plan.json')
    for relative,known in plan['protected_sha256'].items():
        assert sha(R/relative)==known,relative
    assert read(O/'early-stop-audit.json')['epoch']==372
    assert read(O/'train-to-500-exit.json')['returncode']==0
    assert shutil.disk_usage(R).free > 25*1024**3
    return plan


def wait_free(gpus):
    while True:
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        busy={line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
        free={int(parts[0]) for line in devices.splitlines() if len(parts:=[x.strip() for x in line.split(',')])==3 and parts[1] not in busy and int(parts[2])<512}
        if set(gpus)<=free:
            return
        status('waiting_for_free_gpu',requested=gpus)
        time.sleep(30)


def decode_command(epoch, ids, output, full=False):
    config=read(O/'effective-config.json')
    command=[str(PY),'-B',str(R/'decode_brain_adapter.py'),'--no_selection','--split','val',
             '--image_ids',str(ids),'--decoded_stimuli_dir',str(output),'--model_weights_dir',str(D),
             '--saved_epochs',str(epoch),'--num_predictions','2','--noise_factor','4','--precision','fp32',
             '--batch_size','1','--denoising_steps','50','--seed','42','--num_workers','0']
    for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk',
              'condition_dim','num_decoder_queries','sub_approach']:
        command += ['--'+k,str(config[k])]
    if not full:
        command += ['--export_path',str(D/f'exports/epoch-{epoch}/inference.pt')]
    return command


def execute(jobs):
    """Independent processes on distinct GPUs; retain logs and artifacts on failure."""
    assert len({j[1] for j in jobs})==len(jobs)
    wait_free([j[1] for j in jobs])
    frozen()
    children=[]
    for name,gpu,command in jobs:
        assert not (A/(name+'-started.json')).exists()
        log=(A/(name+'.log')).open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
        child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(A/(name+'-started.json'),dict(pid=child.pid,gpu=gpu,command=command,time=time.time()))
        children.append((name,child,log))
    while any(p.poll() is None for _,p,_ in children):
        status('running',stages=[dict(name=n,pid=p.pid,returncode=p.poll()) for n,p,_ in children])
        time.sleep(15)
    codes=[]
    for name,child,log in children:
        log.close()
        write(A/(name+'-exit.json'),dict(pid=child.pid,returncode=child.returncode,time=time.time()))
        codes.append(child.returncode)
    assert all(x==0 for x in codes), f'Failed stage: {[(n,p.returncode) for n,p,_ in children]}'


def seed_for(i,c):
    return int.from_bytes(hashlib.sha256(f'42:{i}:{c}'.encode()).digest()[:8],'little')%(2**63-1)


def validate_generation(epoch):
    import numpy as np
    import h5py
    ids=np.loadtxt(IDS,dtype=np.int64).reshape(-1)
    base=A/f'epoch-{epoch}'
    combined=base/'decoded'
    assert not combined.exists()
    combined.mkdir()
    files=[]
    for part in ['gpu0','gpu1']:
        files.extend((base/part).glob('image_*.npz'))
    if epoch==109:
        files.extend((A/'preflight-export').glob('image_*.npz'))
    assert len(files)==500 and len({p.name for p in files})==500
    for source in files:
        os.link(source,combined/source.name)
    config=read(O/'effective-config.json')
    records=[]
    with h5py.File(Path(config['imgs_dir'])/'nsd_stimuli.hdf5','r') as gtfile:
        for i in ids:
            i=int(i);path=combined/f'image_{i:06d}.npz'
            with np.load(path) as z:
                assert int(z['image_id'])==i and z['candidate_images'].shape==(2,512,512,3)
                assert z['candidate_images'].dtype==np.uint8
                expected=[seed_for(i,c) for c in range(2)]
                assert z['seeds'].tolist()==expected and len(set(expected))==2
                gt=gtfile['imgBrick'][i].transpose(2,0,1).astype(np.float32)/255
                assert np.array_equal(gt,z['groundtruth_image'])
                records.append(dict(image_id=i,seeds=expected,npz_sha256=sha(path),
                                    pixel_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in z['candidate_images']]))
    write(base/'generation-audit.json',dict(n=500,K=2,records=records,groundtruth_matches_source=True,
                                          selection=False,training_updates=0))


def score(epoch):
    frozen()
    import torch
    import numpy as np
    import pandas as pd
    torch.set_num_threads(4)
    sys.path.insert(0,str(R/'scripts'))
    sys.path.insert(0,str(R/'benchmarks/subj01-v1'))
    import evaluate_stage as ev
    import metric_brain_adapter as m
    from metric_feature_batches import install
    install(m,16)
    base=A/f'epoch-{epoch}'
    generation=read(base/'generation-audit.json')
    for item in generation['records']:
        assert sha(base/'decoded'/f'image_{item["image_id"]:06d}.npz')==item['npz_sha256']
    assert not (base/'val500').exists()
    result=ev.metrics(base/'decoded',IDS,base/'val500')
    assert result['n']==500 and result['candidate_selection'] is False
    for candidate in range(2):
        frame=pd.read_csv(base/'val500'/f'candidate-{candidate}.csv')
        assert frame['image_id'].tolist()==np.loadtxt(IDS,dtype=int).tolist()
        for name,value in result['candidate_metrics'][candidate].items():
            assert np.isfinite(frame[name]).all() and np.isclose(frame[name].mean(),value,rtol=0,atol=1e-10)
    ev.atlas(base/'decoded',R/'data/subject02/probe_ids.txt',base/'atlas')
    write(base/'scoring-audit.json',dict(n=500,K=2,feature_batch=16,negative_pool=500,
                                       per_image_means_verified=True,metric_formula_unchanged=True,
                                       candidate_selection=False,visual_review_pending=True))


def preflight():
    import numpy as np
    ids=np.loadtxt(IDS,dtype=np.int64)
    path=A/'preflight-ids.txt'
    assert not path.exists();np.savetxt(path,ids[:2],fmt='%d')
    execute([('preflight-export',0,decode_command(109,path,A/'preflight-export')),
             ('preflight-full',1,decode_command(109,path,A/'preflight-full',True))])
    checks=[]
    for i in ids[:2]:
        with np.load(A/'preflight-export'/f'image_{i:06d}.npz') as a,np.load(A/'preflight-full'/f'image_{i:06d}.npz') as b:
            assert np.array_equal(a['candidate_images'],b['candidate_images'])
            assert a['seeds'].tolist()==b['seeds'].tolist()==[seed_for(int(i),c) for c in range(2)]
            assert np.array_equal(a['groundtruth_image'],b['groundtruth_image'])
            checks.append(dict(image_id=int(i),pixels_exact=True,seeds_match=True))
    write(A/'preflight.json',dict(passed=True,checks=checks,export_equals_full_checkpoint=True,training_updates=0))


def comparison():
    import numpy as np
    from PIL import Image,ImageDraw
    sys.path.insert(0,str(R/'scripts'))
    import balanced_controller as original
    metrics={epoch:read(A/f'epoch-{epoch}/val500/metrics.json') for epoch in EPOCHS}
    gates={epoch:original.gate(metrics[epoch],metrics[109]) for epoch in (300,372)}
    eligible=[(e,metrics[e]) for e in (300,372) if gates[e]['passed']]
    frontier=[]
    for e,m in eligible:
        v=[m['S'],m['mean']['PixCorr'],m['mean']['SSIM']]
        if not any(all(x>=y for x,y in zip([other['S'],other['mean']['PixCorr'],other['mean']['SSIM']],v)) and
                   any(x>y for x,y in zip([other['S'],other['mean']['PixCorr'],other['mean']['SSIM']],v)) for oe,other in eligible if oe!=e):
            frontier.append((e,m))
    candidate=sorted(frontier,key=lambda x:(-x[1]['S'],-x[1]['mean']['PixCorr'],-x[1]['mean']['SSIM'],x[0]))[0][0] if frontier else 109
    write(A/'k2-comparison.json',dict(metrics={str(e):metrics[e]['mean'] for e in EPOCHS},
                                    S={str(e):metrics[e]['S'] for e in EPOCHS},gates=gates,
                                    candidate_epoch=candidate,winner_confirmed=False,
                                    K4_confirmation_required=candidate!=109,visual_review_pending=True,
                                    test_not_used=True,posthoc_checkpoint_comparison=True))
    atlas=A/'comparison-atlas';atlas.mkdir()
    ids=np.loadtxt(R/'data/subject02/probe_ids.txt',dtype=int)
    for page in range(4):
        canvas=Image.new('RGB',(1024,40+8*280),'white');draw=ImageDraw.Draw(canvas)
        for k,label in enumerate(['GT','Epoch109 K1','Epoch300 K1','Epoch372 K1']):draw.text((k*256+8,10),label,fill='black')
        for row,i in enumerate(ids[page*8:(page+1)*8]):
            files=[np.load(A/f'epoch-{e}'/'decoded'/f'image_{i:06d}.npz') for e in EPOCHS]
            images=[(files[0]['groundtruth_image'].transpose(1,2,0)*255).round().astype('uint8')]+[f['candidate_images'][0] for f in files]
            for column,img in enumerate(images):canvas.paste(Image.fromarray(img).resize((256,256)),(column*256,40+row*280))
            draw.text((8,40+row*280+258),f'ID {i}; fixed candidate0; no image selection',fill='black')
            for f in files:f.close()
        canvas.save(atlas/f'page-{page+1}.png')


def controller():
    import numpy as np
    lock=(A/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (A/'started.json').exists(),'Explicit inspected recovery required'
    frozen()
    write(A/'started.json',dict(pid=os.getpid(),time=time.time()))
    preflight()
    ids=np.loadtxt(IDS,dtype=np.int64)
    for epoch in EPOCHS:
        base=A/f'epoch-{epoch}';base.mkdir()
        remaining=ids[2:] if epoch==109 else ids
        jobs=[]
        for gpu in (0,1):
            subset=base/f'gpu{gpu}-ids.txt';np.savetxt(subset,remaining[gpu::2],fmt='%d')
            jobs.append((f'epoch{epoch}-decode-gpu{gpu}',gpu,decode_command(epoch,subset,base/f'gpu{gpu}')))
        execute(jobs)
        validate_generation(epoch)
        execute([(f'epoch{epoch}-score',0,[str(PY),'-B',str(Path(__file__).resolve()),'score','--epoch',str(epoch)])])
        frozen()
    comparison()
    status('validation_complete_needs_visual_review',epochs=list(EPOCHS),training_stopped=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['controller','score']);p.add_argument('--epoch',type=int,choices=EPOCHS);a=p.parse_args()
    try:
        if a.mode=='controller':controller()
        else:score(a.epoch)
    except Exception:
        target=A/('failure.json' if a.mode=='controller' else f'epoch{a.epoch}-score-failure.json')
        if not target.exists():write(target,dict(time=time.time(),traceback=traceback.format_exc()))
        if a.mode=='controller':status('failed_needs_inspection',failure=str(target))
        raise
