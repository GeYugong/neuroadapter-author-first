"""Validate pinned Subject2 encoder identity on CPU; GPU equivalence is separate."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path,PosixPath
import os
import time
import traceback

R=Path('/data1/matengyu/geyugong/neuroadapter-author-first')
O=R/'reports/subject02-bf16-seed42/final-test-preparation'
DEST=R/'runtime/subject02-brain-encoder'


def write(name,data):
    p=O/name;tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    lock=(O/'cpu-audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'encoder-assets-ready.json').exists()
    start=time.time()
    while len(list((O/'encoder-downloads').glob('*.json')))<32:
        write('cpu-audit-status.json',dict(state='waiting_for_downloads',pid=os.getpid(),time=time.time(),verified_files=len(list((O/'encoder-downloads').glob('*.json')))))
        assert time.time()-start<6*3600,'Download incomplete after six hours; inspect local stream'
        time.sleep(30)
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.serialization.add_safe_globals([argparse.Namespace,PosixPath])
    plan=json.loads((O/'encoder-download-plan.json').read_text())
    members=[];correlations=[]
    for index,item in enumerate(plan['files']):
        p=DEST/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
        receipt=json.loads((O/'encoder-downloads'/f'{index:02}.json').read_text());assert receipt['sha256']==item['sha256'] and receipt['verified']
        if p.suffix=='.npy':
            a=np.load(p,allow_pickle=False);assert a.shape==(163842,)
            correlations.append(dict(path=item['path'],sha256=item['sha256'],dtype=str(a.dtype),shape=list(a.shape),nan_count=int(np.isnan(a).sum()),inf_count=int(np.isinf(a).sum())))
            continue
        checkpoint=torch.load(p,map_location='cpu',weights_only=True);args=checkpoint['args'];state=checkpoint['model']
        layer=int(p.parents[2].name.split('_')[1]);folder_run=int(p.parents[1].name.split('_')[1]);hemi=p.parent.name
        assert int(args.subj)==2 and args.hemi==hemi and args.enc_output_layer==layer
        assert args.backbone_arch=='dinov2_q' and args.encoder_arch=='transformer'
        h=hashlib.sha256()
        for key,value in sorted(state.items()):
            assert isinstance(value,torch.Tensor) and torch.isfinite(value).all(),key
            h.update(json.dumps([key,str(value.dtype),list(value.shape)]).encode())
            h.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        members.append(dict(path=item['path'],file_sha256=item['sha256'],model_tensors_sha256=h.hexdigest(),subject=2,hemi=hemi,layer=layer,folder_run=folder_run,checkpoint_args_run=args.run,epoch=int(checkpoint['epoch']),safe_weights_only_load=True,tensor_count=len(state)))
        write('cpu-audit-status.json',dict(state='auditing_members',pid=os.getpid(),time=time.time(),completed=len(members),total=16))
        del checkpoint,state
    assert len(members)==len(correlations)==16 and len({m['model_tensors_sha256'] for m in members})==16
    assert {(m['hemi'],m['layer'],m['folder_run']) for m in members}=={(h,l,n) for h in ['lh','rh'] for l in [1,3,5,7] for n in [1,2]}
    result=dict(subject=2,complete=True,files=32,total_bytes=plan['total_bytes'],repository=plan['repository'],revision=plan['revision'],members=members,correlations=correlations,all_official_LFS_hashes_match=True,all_model_tensors_finite=True,ensemble_model_states_distinct=True,metadata_note='Folder run1/2 identifies published ensemble members; checkpoint args.run is a different author experiment label. Preserve both; do not silently relabel it.',correlation_policy='Preserve downloaded arrays; author inference uses nan_to_num then scipy float64 softmax20',full_forward_equivalence_pending=True,encoder_development_split_verified=False,training_updates=0,gpu_used=False,time=time.time())
    write('encoder-assets-ready.json',result)
    write('cpu-audit-status.json',dict(state='assets_ready_gpu_preflight_pending',pid=os.getpid(),time=time.time(),files=32,members=16))
    print(json.dumps({'assets_ready':True,'members':16,'gpu_preflight_pending':True}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        write('cpu-audit-failure.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
