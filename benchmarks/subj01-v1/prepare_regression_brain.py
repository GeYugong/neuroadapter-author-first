"""CPU-only native regression brain input preparation; no model training or GPU."""
import argparse,datetime,gc,hashlib,io,json,os,time
from pathlib import Path
import h5py,nibabel as nib,numpy as np,pandas as pd,requests,scipy.io
ROOT=Path('/data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1');ASSETS=ROOT/'assets/nsd-func1pt8';OUT=ROOT/'processed/regression-brain'
EXP=Path('/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd/experiments/nsd/nsd_expdesign.mat')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def savej(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');os.replace(q,p)
class Remote(io.RawIOBase):
 def __init__(self,url):
  self.s=requests.Session();self.s.trust_env=False;self.url=url;self.pos=0;self.bytes=0
  r=self.s.head(url,timeout=30);r.raise_for_status();self.size=int(r.headers['Content-Length']);self.etag=r.headers['ETag']
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,n,w=0):self.pos=n if w==0 else self.pos+n if w==1 else self.size+n;return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  if n==0:return b''
  r=self.s.get(self.url,headers={'Range':f'bytes={self.pos}-{min(self.size,self.pos+n)-1}','If-Match':self.etag},timeout=60);r.raise_for_status()
  if r.status_code!=206:raise RuntimeError('Range required')
  self.pos+=len(r.content);self.bytes+=len(r.content);return r.content
 def readinto(self,b):v=self.read(len(b));b[:len(v)]=v;return len(v)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait-download',action='store_true');args=ap.parse_args()
 marker=ASSETS/'DOWNLOAD_COMPLETE.json'
 deadline=time.time()+7200
 while not marker.exists():
  if not args.wait_download or time.time()>deadline:raise RuntimeError('Download completion required')
  print(json.dumps({'event':'waiting_for_download','utc':now()}),flush=True);time.sleep(30)
 if (OUT/'COMPLETE.json').exists():
  print('Already complete; no rewrite',flush=True);return
 if OUT.exists():raise RuntimeError('Existing incomplete output; inspect explicitly before resuming')
 OUT.mkdir(parents=True)
 savej(OUT/'STARTED.json',{'utc':now(),'pid':os.getpid(),'script_sha256':sha(Path(__file__)),'download_marker_sha256':sha(marker),'command':'runtime/env/bin/python benchmarks/subj01-v1/prepare_regression_brain.py --wait-download'})
 download=json.loads(marker.read_text());checks={x['path']:x for x in download['files']}
 for x in download['files']:
  p=Path(x['path'])
  if p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise RuntimeError('Asset hash mismatch '+str(p))
 d=scipy.io.loadmat(EXP,squeeze_me=True);order=np.asarray(d['masterordering']).reshape(-1)[:27750].astype(int);trial_ids=np.asarray(d['subjectim'])[0,order-1].astype(np.int64)-1
 beh=pd.read_csv(ASSETS/'nsddata/ppdata/subj01/behav/responses.tsv',sep='\t');observed=np.concatenate([beh.loc[beh.SESSION==i,'73KID'].to_numpy(dtype=np.int64)-1 for i in range(1,38)])
 assert np.array_equal(trial_ids,observed)
 shared=set((np.asarray(d['sharedix']).reshape(-1)-1).astype(int).tolist());is_test=np.array([int(i) in shared for i in trial_ids]);assert np.array_equal(is_test,order<=1000)
 train_ids=np.array(list(dict.fromkeys(trial_ids[~is_test].tolist())));test_ids=np.array(list(dict.fromkeys(trial_ids[is_test].tolist())))
 train_groups=[np.flatnonzero(trial_ids==i) for i in train_ids];test_groups=[np.flatnonzero(trial_ids==i) for i in test_ids]
 for name,arr in [('all_trial_ids',trial_ids),('train_unique_ids',train_ids),('test_unique_ids',test_ids),('train_trial_ids',trial_ids[~is_test]),('test_trial_ids',trial_ids[is_test])]:np.save(OUT/(name+'.npy'),arr)
 roi=ASSETS/'nsddata/ppdata/subj01/func1pt8mm/roi';general=nib.load(roi/'nsdgeneral.nii.gz').get_fdata()>0;streams=nib.load(roi/'streams.nii.gz').get_fdata();labels=pd.read_csv(ASSETS/'nsddata/freesurfer/fsaverage/label/streams.mgz.ctab',sep=r'\s+',header=None);names={str(row[1]):int(row[0]) for row in labels.values};masks={name:streams.transpose(2,1,0)==names[name] for name in ('early','ventral')}
 beta_dir=ASSETS/'nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR'
 # Validate actual author HDF5 layout/value for fixed first/middle/last ROI positions.
 first=nib.load(beta_dir/'betas_session01.nii.gz');raw=np.asanyarray(first.dataobj)
 assert raw.dtype==np.int16 and first.dataobj.slope==1 and first.dataobj.inter==0
 url='https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.hdf5';remote=Remote(url);native_checks=[]
 with h5py.File(remote,'r') as f:
  b=f['betas'];assert b.shape==(750,83,104,81) and b.dtype==np.int16
  for name,mask in masks.items():
   coords=np.argwhere(mask)
   for idx in [0,len(coords)//2,len(coords)-1]:
    z,y,x=map(int,coords[idx]);reference=b[:,z,y,x];candidate=raw[x,y,z,:]
    assert np.array_equal(reference,candidate)
    native_checks.append({'roi':name,'zyx':[z,y,x],'n_trials':750,'exact':True,'reference_sha256':hashlib.sha256(reference.tobytes()).hexdigest()})
 # Original Brain-Diffuser get_fdata(float64).astype(float32) versus ROI-first conversion.
 reference=first.get_fdata(caching='unchanged')[general].T.astype(np.float32)
 candidate=raw[general].T.astype(np.float32)
 assert np.array_equal(reference,candidate)
 avg_checks=[]
 for group in [np.array([0]),np.array([0,1]),np.array([0,1,2])]:
  assert np.array_equal(reference[group].mean(0),candidate[group].mean(0));avg_checks.append({'trial_indices':group.tolist(),'float32_mean_exact':True})
 del reference,candidate;gc.collect()
 savej(OUT/'preflight.json',{'utc':now(),'official_Takagi_HDF5_url':url,'official_HDF5_etag':remote.etag,'range_bytes_read':remote.bytes,'HDF5_nifti_exact_checks':native_checks,'BrainDiffuser_full_first_session_ROI_float32_exact':True,'BrainDiffuser_mean_checks':avg_checks,'raw_dtype':'int16','slope':1,'inter':0,'brain_diffuser_mask_order':'C-order boolean selection x,y,z','takagi_mask_order':'C-order boolean selection z,y,x after transpose(3,2,1,0)','brain_diffuser_scaling':'No /300 here; author regression performs /300 then train std(ddof1)','takagi_scaling':'Raw int16 as original HDF5; Ridge later casts float32 and StandardScaler'})
 bd=np.lib.format.open_memmap(OUT/'bd_trials.npy',mode='w+',dtype=np.float32,shape=(27750,int(general.sum())))
 tk={name:np.lib.format.open_memmap(OUT/('takagi_'+name+'_trials.npy'),mode='w+',dtype=np.int16,shape=(27750,int(mask.sum()))) for name,mask in masks.items()}
 for session in range(1,38):
  if session!=1:
   image=nib.load(beta_dir/f'betas_session{session:02d}.nii.gz');assert image.dataobj.slope==1 and image.dataobj.inter==0;raw=np.asanyarray(image.dataobj)
  assert raw.shape==(81,104,83,750) and raw.dtype==np.int16
  section=slice((session-1)*750,session*750)
  bd[section]=raw[general].T.astype(np.float32)
  native=raw.transpose(3,2,1,0)
  for name,mask in masks.items():tk[name][section]=native[:,mask]
  bd.flush()
  for arr in tk.values():arr.flush()
  del native,raw;gc.collect()
  print(json.dumps({'event':'session_extracted','session':session,'utc':now()}),flush=True)
 outputs=[]
 def record(p):
  a=np.load(p,mmap_mode='r');outputs.append({'path':str(p),'shape':list(a.shape),'dtype':str(a.dtype),'sha256':sha(p)});del a
 for split,groups in [('train',train_groups),('test',test_groups)]:
  p=OUT/f'nsd_{split}_fmriavg_nsdgeneral_sub1.npy';out=np.lib.format.open_memmap(p,mode='w+',dtype=np.float64,shape=(len(groups),bd.shape[1]))
  for i,g in enumerate(groups):out[i]=bd[g].mean(0)
  out.flush();del out;record(p)
 for name,data in tk.items():
  p=OUT/f'subj01_{name}_betas_tr.npy';out=np.lib.format.open_memmap(p,mode='w+',dtype=np.int16,shape=(int((~is_test).sum()),data.shape[1]));n=0
  for session in range(37):
   sl=slice(session*750,(session+1)*750);block=data[sl][~is_test[sl]];out[n:n+len(block)]=block;n+=len(block)
  out.flush();del out;record(p)
  p=OUT/f'subj01_{name}_betas_ave_te.npy';out=np.lib.format.open_memmap(p,mode='w+',dtype=np.float64,shape=(len(test_groups),data.shape[1]))
  for i,g in enumerate(test_groups):out[i]=data[g].mean(0)
  out.flush();del out;record(p)
 for p in OUT.glob('*ids.npy'):record(p)
 savej(OUT/'COMPLETE.json',{'utc':now(),'exit_status':0,'n_train_unique':len(train_ids),'n_test_unique':len(test_ids),'n_train_trials':int((~is_test).sum()),'n_test_trials':int(is_test.sum()),'outputs':outputs,'source_expdesign_sha256':sha(EXP),'code_sources':{'brain_diffuser_prepare_sha256':sha(ROOT/'upstream/brain-diffuser/data/prepare_nsddata.py'),'takagi_make_subjmri_sha256':sha(ROOT/'upstream/takagi/codes/utils/make_subjmri.py'),'takagi_embedded_nsda_sha256':sha(ROOT/'upstream/takagi/codes/utils/nsd_access/nsda.py')},'IO_changes_only':['Sequential per-session ROI extraction instead of all-volume concatenation','Official HDF5 equivalence checked against actual byte-range values; NIfTI int16 transpose used','Memory-mapped trial arrays and rowwise author means; no images loaded','No training, normalization, target-feature extraction or GPU work performed']})
 print(json.dumps({'event':'complete','utc':now(),'exit_status':0}),flush=True)
if __name__=='__main__':main()
