"""MindHier retrieval on the shared S1 IDs; source project remains read-only."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
import sys,json,time,traceback,argparse,gc
import numpy as np
import torch,h5py
from retrieval_common import R,O,sha,digest,write,evaluate,arithmetic_mean,CommonClip
M=Path('/data/matengyu/geyugong/mindhier-reproduction')
ids=np.load(O/'ids-s1.npy');draws=np.load(O/'draws-s1.npy')
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.09);os.nice(10)
def records(folder):
 data=[json.loads(x) for x in (folder/'samples.jsonl').read_text().splitlines()]
 assert len(data)==1000 and len({x['nsd_id'] for x in data})==1000
 lookup={x['nsd_id']:x for x in data};assert set(map(int,ids))<=set(lookup)
 return [lookup[int(i)] for i in ids]
def progress(mode,n):
 (O/f'mindhier-{mode}-status.json').write_text(json.dumps(dict(completed=n,total=982,time=time.time())))
def native():
 sys.path.insert(0,str(M/'src/mindhier/MindHier'))
 from models.fmri_c2f_encoder import FMRI2CLIP
 path=M/'runs/s01-stage1-paper-seed42/latest.pt';weight_hash=sha(path)
 ck=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 assert ck['step']==15600 and ck['next_epoch']==300
 model=FMRI2CLIP(input_dim=15724,multi_subj=False)
 model.load_state_dict(ck['model_state_dict'],strict=True);del ck
 model.eval().requires_grad_(False).cuda()
 rows=records(M/'artifacts/s01-final8500-n1');parts=[]
 with h5py.File(M/'data/nsd/betas_all_subj01_fp32_renorm.hdf5','r') as f,torch.inference_mode():
  for start in range(0,982,4):
   x=torch.cat([torch.from_numpy(np.stack([f['betas'][j] for j in rec['beta_rows']])).mean(0,keepdim=True) for rec in rows[start:start+4]])
   y=model(x.cuda())[0].float().cpu().numpy();assert y.shape[1:]==(257,768)
   parts.append(y)
   if start%100==0:progress('native',start+len(y));print('NATIVE',start,flush=True)
 a=np.concatenate(parts);del model;torch.cuda.empty_cache();gc.collect()
 target_dir=M/'data/clip_targets_subj01'
 allids=np.load(target_dir/'image_ids.npy');lookup={int(x):i for i,x in enumerate(allids)}
 target=np.load(target_dir/'image_layer_24.npy',mmap_mode='r')
 b=np.stack([target[lookup[int(i)]] for i in ids])
 result=evaluate(a,b,draws,O/'native/mindhier_s1',dict(method='mindhier_s1',subject=1,kind='native',
  direction_names=['Image (brain-to-image)','Brain (image-to-brain)'],
  feature='Stage1 projected_image_emb and matching cached projected CLIP-L/14 hidden layer24 tokens;257x768',
  checkpoint_epoch=300,checkpoint_step=15600,checkpoint_sha256=weight_hash,
  target_preprocessing='Native PIL LANCZOS512 and CLIPImageProcessor do_rescale=False; existing FP32 cache originally made with TF32 enabled',
  input='Same three beta rows and FP32 arithmetic mean as fixed8500 reconstruction',
  source_sha256={str(Path(__file__)):sha(__file__),str(M/'src/mindhier/MindHier/models/fmri_c2f_encoder.py'):sha(M/'src/mindhier/MindHier/models/fmri_c2f_encoder.py'),str(M/'scripts/cache_clip.py'):sha(M/'scripts/cache_clip.py')},
  precision='FP32 model forward on RTX5090 with TF32 disabled; native cached target FP32; FP64 cosine',
  original_test_count=1000,evaluated_common_ids=982))
 assert sha(path)==weight_hash
 write(O/'mindhier-native-exit.json',dict(returncode=0,time=time.time(),mean=result['mean_percent']))
def images():
 dest=O/'reconstruction/mindhier_s1';assert not dest.exists();dest.mkdir(parents=True)
 cp=M/'pretrained/metrics/ViT-L-14.pt'
 expected=json.loads((O/'common-clip-source.json').read_text())
 assert sha(cp)==expected['sha256']
 ex=CommonClip(cp,batch=4)
 with h5py.File(M/'data/nsd_stimuli.hdf5','r') as h:
  x=torch.stack([torch.from_numpy(h['imgBrick'][int(i)]).permute(2,0,1).float()/255 for i in ids[:4]])
 features=ex(x);gt=np.load(O/'gt-common-s1.npy')
 error=float(np.max(np.abs(features-gt[:4])));assert np.allclose(features,gt[:4],atol=2e-4,rtol=2e-4),error
 write(dest/'cross-host-sanity.json',dict(common_CLIP_weight_sha256=sha(cp),first4_GT_features_max_error=error,matching_preprocessing=True,reference_host='4090',execution_host='5090'))
 summaries={}
 for label,k in [('n1',1),('n4',4)]:
  folder=M/f'artifacts/s01-final8500-{label}';rows=records(folder);parts=[];inputs=[];picks=[]
  selected=records(M/'artifacts/s01-final8500-n4-selected') if k==4 else None
  for pos,row in enumerate(rows):
   p=folder/row['tensor'];assert sha(p)==row['sha256']
   tensor=torch.load(p,map_location='cpu',weights_only=True).float();assert tensor.shape==(k,3,512,512)
   if k==4:
    sr=selected[pos];assert sr['nsd_id']==row['nsd_id'] and sr['source_tensor_sha256']==row['sha256']
    sp=M/'artifacts/s01-final8500-n4-selected'/sr['tensor'];assert sha(sp)==sr['sha256']
    st=torch.load(sp,map_location='cpu',weights_only=True).float()
    assert torch.equal(st.reshape(1,3,512,512)[0],tensor[sr['selected_index']])
    assert int(np.argmax(sr['selection_scores']))==sr['selected_index']
    picks.append(sr['selected_index'])
   parts.append(ex(tensor));inputs.append(dict(image_id=row['nsd_id'],path=str(p),sha256=row['sha256']))
   if (pos+1)%50==0:progress(label,pos+1);print(label,pos+1,flush=True)
  f=np.stack(parts);np.savez(dest/f'{label}-features.npz',ids=ids,features=f,picks=picks)
  with (dest/f'{label}-inputs.jsonl').open('x') as log:
   for rec in inputs:log.write(json.dumps(rec)+'\n')
  rr=[]
  for j in range(k):
   rr.append(evaluate(f[:,j],gt,draws,dest/f'{label}_candidate{j}',dict(method='mindhier_s1',subject=1,kind='reconstruction',variant=f'{label}_candidate{j}',GT_assisted=False,direction_names=['Recon-to-Image','Image-to-Recon'],original_test_count=1000,evaluated_common_ids=982)))
  if k==4:
   evaluate(f[np.arange(982),picks],gt,draws,dest/'n4_selected',dict(method='mindhier_s1',subject=1,kind='reconstruction',variant='N4_GT_assisted',GT_assisted=True,direction_names=['Recon-to-Image','Image-to-Recon']))
   arithmetic_mean(rr,dest/'n4_mean',dict(method='mindhier_s1',subject=1,kind='reconstruction',variant='N4_mean',GT_assisted=False,direction_names=['Recon-to-Image','Image-to-Recon']))
 write(dest/'complete.json',dict(complete=True,common_ids=982,n1_candidates=982,n4_candidates=3928,time=time.time()))
 write(O/'mindhier-images-exit.json',dict(returncode=0,time=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['native','images']);a=p.parse_args()
 try:native() if a.mode=='native' else images()
 except BaseException:
  write(O/f'mindhier-{a.mode}-failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
