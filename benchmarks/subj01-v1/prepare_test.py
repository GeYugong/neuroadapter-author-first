from pathlib import Path
import csv,hashlib,io,json,tarfile
import numpy as np
import pandas as pd
import scipy.io
import h5py,torch
from PIL import Image
B=Path(__file__).resolve().parent; R=B.parents[1]
raw=Path('/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd')
lock=json.loads((B/'mindeye-download-lock.json').read_text())
rows=[]; brain=[]; ids=[]
for f in lock['files']:
 if '/test/test_subj01_' not in f['path']:continue
 p=B/'assets/mindeye'/f['path'];h=hashlib.sha256()
 with p.open('rb') as stream:
  for c in iter(lambda:stream.read(8388608),b''):h.update(c)
 assert h.hexdigest()==f['lfs_sha256'],p
 with tarfile.open(p,'r:') as tf:
  records={}
  for member in tf:
   key,ext=member.name.split('.',1)
   if ext not in ('coco73k.npy','nsdgeneral.npy','num_uniques.npy','trial.npy','jpg'):continue
   records.setdefault(key,{})[ext]=member.name
  for index,(key,names) in enumerate(records.items()):
   def arr(name):return np.load(io.BytesIO(tf.extractfile(names[name]).read()),allow_pickle=False)
   nsdid=int(arr('coco73k.npy').item());v=arr('nsdgeneral.npy')
   assert v.shape==(3,15724) and v.dtype==np.float16
   # Exact author operation: mean the three float16 repeats, THEN float32.
   mean=torch.from_numpy(v).mean(dim=0).float().numpy()
   ids.append(nsdid);brain.append(mean)
   rows.append(dict(nsd_id0=nsdid,mindeye_key=key,shard=f['path'],source_index=index,trial=int(arr('trial.npy').item()),num_uniques=int(arr('num_uniques.npy').item()),jpeg_member=names['jpg']))
assert len(set(ids))==len(ids)
meta=pd.read_csv(raw/'experiments/nsd/nsd_stim_info_merged.csv').set_index('nsdId')
shared=set((scipy.io.loadmat(raw/'experiments/nsd/nsd_expdesign.mat')['sharedix'].reshape(-1)-1).astype(int))
na_test=set(np.loadtxt(R/'configs/test_ids.txt',dtype=int).tolist())
na_train=set(np.loadtxt(R/'configs/train_ids.txt',dtype=int).tolist()); na_val=set(np.loadtxt(R/'configs/val_ids.txt',dtype=int).tolist())
assert set(ids)<=shared, 'coco73k IDs are not verified zero-based NSD shared IDs'
T=sorted(set(ids)&na_test);assert not(set(T)&(na_train|na_val))
assert all(meta.loc[i,'shared1000'] and meta.loc[i,'subject1'] for i in T)
ordered=sorted(T,key=lambda i:hashlib.sha256(('nsd-benchmark-v1:'+str(i)).encode()).hexdigest())
T100=ordered[:100];atlas=T100[:32]
assert len(T100)==100
B.joinpath('assets/gt').mkdir(exist_ok=True)
lookup={row['nsd_id0']:row for row in rows}
manifest=[]
with h5py.File(raw/'stimuli/nsd_stimuli.hdf5','r') as h:
 for i in T:
  gt=np.asarray(h['imgBrick'][i]);assert gt.shape==(425,425,3) and gt.dtype==np.uint8
  out=B/'assets/gt'/f'{i}.png';Image.fromarray(gt).save(out)
  manifest.append(dict(nsd_id0=i,coco_image_id=int(meta.loc[i,'cocoId']),subject=1,split_source='official MindEye1 test coco73k + NSD sharedix + NeuroAdapter test intersection',gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),per_method_source_index=json.dumps({'MindEye1':lookup[i],'NeuroAdapter':int(np.flatnonzero(np.loadtxt(R/'configs/test_ids.txt',dtype=int)==i)[0])}),eligibility_notes='shared test; no NeuroAdapter train/development500 overlap; not globally unseen historical blind test'))
# Freeze only after identity/index checks. JPEG FOV review images are diagnostic, not scoring GT.
checks=[]
for i in T100[:5]:
 row=lookup[i]
 with tarfile.open(B/'assets/mindeye'/row['shard'],'r:') as tf:
  official=Image.open(io.BytesIO(tf.extractfile(row['jpeg_member']).read())).convert('RGB')
 gt=Image.open(B/'assets/gt'/f'{i}.png')
 pair=Image.new('RGB',(850,425));pair.paste(gt,(0,0));pair.paste(official.resize((425,425)),(425,0));pair.save(B/'assets'/f'identity-{i}.jpg')
 a=np.asarray(gt).astype(float);v=np.asarray(official.resize((425,425))).astype(float)
 corr=float(np.corrcoef(a.reshape(-1),v.reshape(-1))[0,1]) # descriptive only: JPEG256 resampling can lower raw425 correlation
 checks.append({'nsd_id0':i,'coco_id':int(meta.loc[i,'cocoId']),'jpeg_vs_original_pixel_corr':corr,'canonical_gt':'original NSD imgBrick uint8','jpeg_only_for_identity_check':True})
for name,values in [('test_full_ids.txt',T),('test100_ids.txt',T100),('atlas32_ids.txt',atlas),('identity5_ids.txt',T100[:5])]:
 content=''.join(str(i)+'\n' for i in values)
 path=B/name
 if path.exists():assert path.read_text()==content
 else:path.write_text(content)
with (B/'test_manifest.csv').open('w') as out:
 w=csv.DictWriter(out,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
np.savez(B/'assets/subject1_test_brain.npz',nsd_id0=np.asarray(ids,dtype=np.int64),voxels=np.stack(brain))
(B/'test_identity_checks.json').write_text(json.dumps({'official_test_n':len(ids),'intersection_n':len(T),'excluded_na_test':sorted(na_test-set(ids)),'five_checks':checks,'mean_precision':'torch mean float16 then float32, matching author Reconstructions.py','test_full_ids_sha256':hashlib.sha256((B/'test_full_ids.txt').read_bytes()).hexdigest()},indent=2))
print('LOCKED',len(T),'TEST IDS',flush=True)
