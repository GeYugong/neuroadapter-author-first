"""Evaluate retained non-NeuroAdapter reconstructions under one frozen CLIP protocol."""
from pathlib import Path
import argparse,os,sys,json,time,traceback,shutil
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from retrieval_common import R,O,sha,digest,write,evaluate,arithmetic_mean,sanity,CommonClip
import numpy as np
B=R/'benchmarks/subj01-v1'
CLIP=R.parent/'neuroadapter-subject1-research/models/evaluation/clip/ViT-L-14.pt'
H5=R.parent/'neuroadapter-subject1-research/data/raw/nsd/stimuli/nsd_stimuli.hdf5'
def status(**d):
 p=O/'reconstruction-status.json';p.write_text(json.dumps(dict(d,time=time.time()),indent=2))
def prepare():
 assert not O.exists(),'Existing run; inspect, do not overwrite'
 O.mkdir()
 ids1=np.loadtxt(B/'test_full_ids.txt',dtype=np.int64)
 ids2=np.loadtxt(R/'data/subject02/standard-test-ids.txt',dtype=np.int64)
 assert len(ids1)==len(set(ids1))==len(ids2)==len(set(ids2))==982
 for subject,ids in [(1,ids1),(2,ids2)]:
  np.save(O/f'ids-s{subject}.npy',ids)
  rng=np.random.RandomState(42);draws=np.stack([rng.choice(982,300,replace=False) for _ in range(30)])
  assert all(len(set(x))==300 for x in draws)
  np.save(O/f'draws-s{subject}.npy',draws)
 sources=[R/'scripts/retrieval_common.py',Path(__file__),R/'metric_brain_adapter.py',CLIP]
 write(O/'plan.json',dict(created=time.time(),excluded=['NeuroAdapter S1','NeuroAdapter S2','NeuroAdapter B joint E75','UMBRAE caption-only outputs'],
  common_image_protocol='OpenAI CLIP ViT-L/14 global embeddings; input native tensors/PNG, prediction resize425 bilinear align_corners=False then Resize224 bilinear antialias=True and CLIP normalization; FP32 extraction and FP64 cosine',
  native_protocol='Each model native brain-predicted image feature and its matching frozen visual target space; separate table',
  orientations={'reconstruction':['Recon-to-Image','Image-to-Recon'],'native':['Image (brain-to-image)','Brain (image-to-brain)']},
  paired_draws='RandomState42;30 without-replacement subsets of300 paired IDs per subject; same saved lists for every method',
  selection='Existing selections only; no candidate reselection. GT-assisted rows flagged.',
  source_sha256={str(p):sha(p) for p in sources},
  ids_and_draws_sha256={p.name:sha(p) for p in O.glob('*.npy')},sanity=sanity(),
  storage='Save compact features, score matrices, records and hashes; no regenerated images or training',
  precision_note='New retrieval protocol fixes FP32 features/FP64 cosine, not bitwise reproduction of the original FP16 notebook',
  candidates='All retained full982 candidate sets; fixed32-only extra candidates excluded from full-pool metrics',
  disk_reserve_gib=25))
def main():
 import torch,h5py
 from PIL import Image
 assert shutil.disk_usage(R).free/2**30>=25
 plan=json.loads((O/'plan.json').read_text())
 for p,v in plan['source_sha256'].items():assert sha(p)==v,p
 for p,v in plan['ids_and_draws_sha256'].items():assert sha(O/p)==v,p
 assert not (O/'reconstruction-started.json').exists()
 os.nice(10);torch.cuda.set_per_process_memory_fraction(.35)
 write(O/'reconstruction-started.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),time=time.time()))
 extractor=CommonClip(CLIP)
 ids={s:np.load(O/f'ids-s{s}.npy') for s in [1,2]};draws={s:np.load(O/f'draws-s{s}.npy') for s in [1,2]}
 gt={}
 with h5py.File(H5,'r') as hf:
  for s in [1,2]:
   parts=[]
   for start in range(0,982,16):
    ims=torch.stack([torch.from_numpy(hf['imgBrick'][int(i)]).permute(2,0,1).float()/255 for i in ids[s][start:start+16]])
    if start==0:
     a=extractor(ims[:4]);c=np.concatenate([extractor(x) for x in ims[:4]])
     err=float(np.max(np.abs(a-c)));assert np.allclose(a,c,atol=2e-4,rtol=2e-4),err
     write(O/f'common-batch-sanity-s{s}.json',dict(max_abs_error=err,batch1_vs4=True))
    parts.append(extractor(ims))
   gt[s]=np.concatenate(parts);np.save(O/f'gt-common-s{s}.npy',gt[s])
 specs=[('mindeye1_s1',1,16),('mindbridge_s1',1,8),('takagi_s1',1,5),
        ('brain_diffuser_s1',1,1),('imagenet_s1',1,1),('mindeye1_s2',2,2)]
 for name,s,kcount in specs:
  dest=O/'reconstruction'/name;assert not dest.exists();dest.mkdir(parents=True)
  features=[];picks=[];manifest=[]
  for pos,ident in enumerate(ids[s]):
   i=int(ident);inputs={}
   if name in ['mindeye1_s1','mindbridge_s1']:
    folder=B/'outputs'/('mindeye1-native16' if name=='mindeye1_s1' else 'mindbridge-native8')
    rp=folder/f'{i}.json';rec=json.loads(rp.read_text());tp=folder/rec['candidates_path']
    assert rec['nsd_id0']==i and rec['K']==kcount
    assert sha(tp)==rec['candidates_sha256']
    images=torch.load(tp,map_location='cpu',weights_only=True)[0].float()
    assert images.shape==(kcount,3,512,512);picks.append(int(rec['candidate']))
    inputs={str(tp):rec['candidates_sha256'],str(rp):sha(rp)}
   elif name=='takagi_s1':
    folder=B/'outputs/takagi-native5';rp=folder/f'{i}.json';rec=json.loads(rp.read_text())
    assert rec['nsd_id0']==i and len(rec['candidates'])==5
    array=[]
    for item in rec['candidates']:
     p=folder/item['path'];assert sha(p)==item['sha256'];inputs[str(p)]=item['sha256']
     with Image.open(p) as im:array.append(torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1).float()/255)
    images=torch.stack(array);inputs[str(rp)]=sha(rp)
   elif name=='brain_diffuser_s1':
    p=B/'outputs/brain-diffuser-full982'/f'{i}.png'
    with Image.open(p) as im:images=torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1).float()[None]/255
    inputs={str(p):sha(p)}
   elif name=='imagenet_s1':
    p=R/'runtime/imagenet-retrieval-local-fixed-v1/retrieved-view-tensors'/f'{i}.npy'
    if not p.exists():
     matches=list((R/'runtime/imagenet-retrieval-local-fixed-v1/retrieved-view-tensors').glob(f'*{i}*'))
     assert len(matches)==1,(i,matches);p=matches[0]
    images=torch.from_numpy(np.load(p).copy()).float()
    if images.ndim==3:images=images[None]
    assert images.shape[1]==3,images.shape
    inputs={str(p):sha(p)}
   else:
    folder=R/'reports/mindeye-subject02-test-20261020'
    rec=json.loads((folder/'records'/f'{i}.json').read_text());p=folder/'outputs'/rec['output_file']
    assert rec['image_id']==i and sha(p)==rec['output_sha256']
    with np.load(p) as z:
     assert int(z['image_id'])==i
     images=torch.from_numpy(np.stack([z['k1'],z['selected']])).float()
     for c,idx in enumerate([0,rec['pick']]):assert digest(images[c].numpy())==rec['candidate_tensor_sha256'][idx]
    inputs={str(p):sha(p)}
   feat=extractor(images);assert feat.shape==(kcount,768)
   features.append(feat);manifest.append(dict(image_id=i,inputs=inputs))
   if (pos+1)%50==0:status(phase='features',method=name,completed=pos+1,total=982);print(name,pos+1,flush=True)
  f=np.stack(features);np.savez(dest/'features.npz',ids=ids[s],features=f,picks=np.asarray(picks,dtype=np.int64))
  with (dest/'inputs.jsonl').open('x') as out:
   for rec in manifest:out.write(json.dumps(rec)+'\n')
  results=[]
  for k in range(kcount):
   label=('K1' if k==0 else 'native16_selected') if name=='mindeye1_s2' else f'candidate{k}'
   meta=dict(method=name,subject=s,kind='reconstruction',variant=label,
    direction_names=plan['orientations']['reconstruction'],GT_assisted=False,
    image_source='Existing retained arrays; no new generation or selection')
   results.append(evaluate(f[:,k],gt[s],draws[s],dest/label,meta))
  if picks:
   selected=f[np.arange(982),picks]
   evaluate(selected,gt[s],draws[s],dest/'selected',dict(method=name,subject=s,kind='reconstruction',
    variant='native16' if name=='mindeye1_s1' else 'native8_GT_assisted',
    direction_names=plan['orientations']['reconstruction'],GT_assisted=name=='mindbridge_s1'))
  if name!='mindeye1_s2' and kcount>1:
   arithmetic_mean(results,dest/'candidate_mean',dict(method=name,subject=s,kind='reconstruction',direction_names=plan['orientations']['reconstruction'],GT_assisted=False))
  write(dest/'complete.json',dict(complete=True,n=982,K=kcount,inputs_count=len(manifest),feature_sha256=sha(dest/'features.npz')))
  print('METHOD_COMPLETE',name,flush=True)
 status(phase='complete',methods=len(specs))
 write(O/'reconstruction-exit.json',dict(returncode=0,time=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args()
 try:prepare() if a.mode=='prepare' else main()
 except BaseException:
  if O.exists():write(O/f'reconstruction-{a.mode}-failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
  raise
