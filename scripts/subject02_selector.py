"""Subject 2 author encoder ensemble and full-forward equivalence preflight."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PosixPath
import subprocess
import sys
import time
import traceback
import types

R=Path(__file__).resolve().parents[1]
O=R/'reports/subject02-bf16-seed42'
P=O/'selector-preflight'
ENC=R.parent/'neuroadapter-subject1-research/repo/vendor/whole_brain_encoder'
BASE=R/'benchmarks/subj01-v1/imagenet-retrieval/local-fixed-v1/encoder.py'
WEIGHTS=R/'runtime/subject02-brain-encoder/dinov2_q_transformer/schaefer/subj_02'
sys.path.insert(0,str(R))
os.environ.update(TORCH_HOME=str(R/'runtime/torch'),MPLCONFIGDIR=str(R/'runtime/subject02-selector-mpl'),
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    P.mkdir(exist_ok=True)
    path=P/name;assert not path.exists(),path
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_selector():
    import numpy as np
    import torch
    from torchvision import transforms
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    torch.serialization.add_safe_globals([argparse.Namespace,PosixPath])
    assets=json.loads((O/'final-test-preparation/encoder-assets-ready.json').read_text())
    assert assets['subject']==2 and assets['complete'] and assets['all_official_LFS_hashes_match']
    download=json.loads((O/'final-test-preparation/encoder-download-plan.json').read_text())
    for entry in download['files']:
        assert sha(R/'runtime/subject02-brain-encoder'/entry['path'])==entry['sha256']
    for member in assets['members']:
        assert member['checkpoint_args_run']=={1:1000,2:1001}[member['folder_run']]
    vertices=torch.load(R/'runs/subject02-author-bf16-seed42/exports/epoch-300/parcel_vertices.pt',weights_only=True)
    masks={h:np.zeros(163842,dtype=bool) for h in ['lh','rh']}
    for h in masks:
        assert len(vertices[h])==100
        masks[h][torch.cat(vertices[h]).numpy()]=True
    P.mkdir(exist_ok=True)
    mask_path=P/'mask.npz'
    if mask_path.exists():
        with np.load(mask_path) as old:
            assert all(np.array_equal(old[h],masks[h]) for h in masks)
    else:np.savez_compressed(mask_path,**masks)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(425),transforms.CenterCrop(425),
                                  transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def prep(images):
        result=[]
        for image in images:
            resized=transform(image);padded=torch.zeros(3,434,434);padded[:,:425,:425]=resized;result.append(padded)
        return torch.stack(result)
    ids=np.loadtxt(R/'data/subject02/val_ids.txt',dtype=int)[:8]
    generation=json.loads((O/'validation-after-stop/epoch-109/generation-audit.json').read_text())
    hashes={x['image_id']:x['npz_sha256'] for x in generation['records']}
    for i in ids:
        assert sha(O/f'validation-after-stop/epoch-109/decoded/image_{i:06d}.npz')==hashes[int(i)]
    sample=np.stack([np.load(O/f'validation-after-stop/epoch-109/decoded/image_{i:06d}.npz')['candidate_images'][0] for i in ids])
    batch=prep(sample)
    sys.path.insert(0,str(BASE.parent))
    patches={
        'for layer in [1, 3, 5, 7]:\n                for run in [1, 2]:':'for run in [1, 2]:\n                for layer in [1, 3, 5, 7]:',
        'self.members, self.audit = [], []':'self.members, self.audit, self.reference = [], [], []',
        'check_batch[:2].to(device)':'check_batch.to(device)',
        "ENC / f'parcels/schaefer/{hemi}_labels_s01.pt'":"ROOT / 'data/subject02/parcels' / f'{hemi}_labels_s02.pt'",
        "map_location='cpu', weights_only=False)":"map_location='cpu', weights_only=True)",
        'assert cfg.enc_output_layer == layer and cfg.run == run and cfg.hemi == hemi':
        "assert cfg.enc_output_layer == layer and cfg.run == {1:1000,2:1001}[run] and cfg.hemi == hemi and int(cfg.subj) == 2",
        'self.audit.append(dict(hemi=hemi':'self.reference.append(expected.cpu().clone())\n                    self.audit.append(dict(subject=2,checkpoint_args_run=cfg.run,hemi=hemi',
        "corr.append(np.load(folder / f'{hemi}_val_corr_nonavg.npy'))":"corr.append(np.nan_to_num(np.load(folder / f'{hemi}_val_corr_nonavg.npy')).astype(np.float64))",
        'self.weights = torch.softmax(20 * torch.from_numpy(np.stack(corr)), dim=0)[:, self.mask.cpu()].to(device)':
        'from scipy.special import softmax\n        self.weights = torch.from_numpy(softmax(20*np.stack(corr),axis=0))[:,self.mask.cpu()]',
        'result = (torch.stack(parts) * self.weights[:, None, :]).sum(dim=0)':
        'result = (torch.stack(parts,dim=1).cpu() * self.weights[None,:,:]).sum(dim=1)',
        "os.environ['MPLCONFIGDIR'] = str(ASSETS / 'mpl')":"os.environ['MPLCONFIGDIR'] = str(ROOT / 'runtime/subject02-selector-mpl')",
    }
    source=BASE.read_text()
    for old,new in patches.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    module=types.ModuleType('subject02_masked_encoder');module.__file__=str(BASE)
    exec(compile(source,str(BASE),'exec'),module.__dict__)
    module.MASK=mask_path;module.WEIGHTS=WEIGHTS;module.ENC=ENC;module.ROOT=R
    halves=[module.HalfEncoder(h,'cuda:0',batch) for h in ['lh','rh']]
    sys.path.insert(0,str(ENC))
    from datasets.nsd import nsd_dataset_custom
    native_ds=nsd_dataset_custom(sample,transform=transform)
    assert torch.equal(torch.stack([native_ds[i][0] for i in range(len(sample))]),batch)
    reference=[];predictions=[];errors=[]
    for half in halves:
        native=(torch.stack(half.reference,dim=1)*half.weights[None,:,:]).sum(dim=1).numpy()
        prediction=half.predict(batch)
        assert np.allclose(native,prediction,atol=2e-4,rtol=2e-4)
        reference.append(native);predictions.append(prediction);errors.append(float(np.max(np.abs(native-prediction))))
    import decode_brain_adapter as d
    from types import SimpleNamespace
    cfg=json.loads((O/'effective-config.json').read_text())
    ds=d.nsd_topk_parcel_dataset(SimpleNamespace(subj=2,hemi=None,backbone_arch='dinov2_q',data_dir=cfg['data_dir'],
        imgs_dir=cfg['imgs_dir'],parcel_dir=cfg['parcel_dir'],tokenizer=None,gen_size=512),split='val',topk=100)
    for h in masks:assert all(torch.equal(x,y) for x,y in zip(vertices[h],ds.get_selected_voxel_indices()[h]))
    def full(parts):
        result={}
        for h,arr in zip(['lh','rh'],parts):
            z=torch.zeros(len(arr),163842,dtype=torch.float64);z[:,torch.from_numpy(masks[h])]=torch.from_numpy(arr);result[h]=z
        return result
    def correlate(gt,pred):
        g=(gt-gt.mean(dim=1,keepdim=True))/(gt.std(dim=1,keepdim=True)+1e-8)
        p=(pred-pred.mean(dim=1,keepdim=True))/(pred.std(dim=1,keepdim=True)+1e-8)
        return ((p@g.to(p.dtype).T).squeeze()/gt.size(1)).numpy()
    lookup={int(i):j for j,i in enumerate(ds.base_dataset.split_presented_imgs)}
    checks=[]
    for i in ids:
        row=ds[lookup[int(i)]];brain={k:row[k][None] for k in ['brain_lh_f','brain_rh_f']}
        gt,p=d.process_brain_data_for_correlation(brain,ds,full(predictions),'cpu')
        _,ref=d.process_brain_data_for_correlation(brain,ds,full(reference),'cpu')
        scores=correlate(gt,p);native_scores=correlate(gt,ref)
        assert np.isfinite(scores).all() and int(scores.argmax())==int(native_scores.argmax())
        checks.append(dict(image_id=int(i),selected=int(scores.argmax()),native_selected=int(native_scores.argmax()),
                           max_correlation_error=float(np.max(np.abs(scores-native_scores)))))
    audit=dict(passed=True,subject=2,source_encoder_sha256=sha(BASE),native_wrapper_sha256=sha(ENC/'brain_encoder_wrapper.py'),
               selector_source_sha256=sha(Path(__file__)),patches=patches,preprocessing_exact=True,
               members=[a for half in halves for a in half.audit],ensemble_max_abs_errors=errors,rank_checks=checks,
               mask_vertices={h:int(masks[h].sum()) for h in masks},candidate_source='fixed first8 validation IDs, candidate0; no test inference',
               ensemble='run1 layers1,3,5,7 then run2 layers1,3,5,7; scipy float64 softmax20',
               original_checkpoint_run_metadata_preserved=True,encoder_development_split_verified=False,
               candidate_ranking_uses_GT_image=False,training_updates=0,peak_gpu_bytes=torch.cuda.max_memory_allocated(),time=time.time())
    if not (P/'audit.json').exists():write('audit.json',audit)
    else:
        previous=json.loads((P/'audit.json').read_text())
        assert previous['passed'] and previous['selector_source_sha256']==audit['selector_source_sha256']
        assert previous['source_encoder_sha256']==audit['source_encoder_sha256']
    for half in halves:half.reference=[]
    return d,ds,prep,halves,full,correlate


def wait_and_preflight():
    import fcntl
    P.mkdir(exist_ok=True)
    lock=(P/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (P/'started.json').exists()
    write('started.json',dict(pid=os.getpid(),time=time.time()))
    while True:
        state=json.loads((O/'validation-after-stop/status.json').read_text())
        if state['phase']=='failed_needs_inspection':raise RuntimeError('Validation failed; inspect before GPU preflight')
        if state['phase']=='validation_complete_needs_visual_review':
            uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()[0]
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
            if uuid not in busy:break
        time.sleep(15)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    load_selector()
    write('exit.json',dict(returncode=0,time=time.time()))


if __name__=='__main__':
    try:wait_and_preflight()
    except Exception:
        write('failure.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
