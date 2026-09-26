import concurrent.futures,datetime,hashlib,json,os,time
from pathlib import Path
import requests
ROOT=Path('/data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1')
DEST=ROOT/'assets/nsd-func1pt8';DEST.mkdir(parents=True,exist_ok=True)
ASSETS=json.loads((ROOT/'brain-diffuser-takagi-assets.json').read_text())['assets']
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def atomic(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
def fetch(a):
 path=DEST/a['path'];path.parent.mkdir(parents=True,exist_ok=True);audit=path.with_suffix(path.suffix+'.download.json');part=path.with_suffix(path.suffix+'.part')
 expected=a['bytes'];etag=a['etag'];s=requests.Session();s.trust_env=False
 if path.exists():
  if path.stat().st_size!=expected:raise RuntimeError('Existing length mismatch '+str(path))
  h=digest(path)
  if audit.exists() and json.loads(audit.read_text())['sha256']!=h:raise RuntimeError('Existing SHA mismatch '+str(path))
  result=dict(a,path=str(path),sha256=h,verified_utc=stamp(),status='complete_reused');atomic(audit,result);return result
 last=None
 for attempt in range(6):
  try:
   offset=part.stat().st_size if part.exists() else 0
   if offset>expected:raise RuntimeError('Partial larger than expected')
   if offset<expected:
    headers={'If-Match':etag}
    if offset:headers['Range']=f'bytes={offset}-'
    with s.get(a['url'],headers=headers,stream=True,timeout=(20,90)) as r:
     r.raise_for_status()
     if r.headers.get('ETag')!=etag:raise RuntimeError('Source ETag changed')
     if offset and (r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Server ignored resume range')
     with part.open('ab' if offset else 'wb') as f:
      for block in r.iter_content(4*1024*1024):
       if block:f.write(block)
   if part.stat().st_size!=expected:raise RuntimeError('Downloaded length mismatch')
   h=digest(part);os.replace(part,path)
   result=dict(a,path=str(path),sha256=h,verified_utc=stamp(),status='complete',attempt=attempt+1);atomic(audit,result)
   print(json.dumps({'event':'file_complete','path':a['path'],'bytes':expected,'sha256':h,'utc':stamp()}),flush=True)
   return result
  except Exception as e:
   last=str(e);print(json.dumps({'event':'retry','path':a['path'],'attempt':attempt+1,'error':last,'utc':stamp()}),flush=True);time.sleep(min(5*(attempt+1),30))
 raise RuntimeError(a['path']+': '+str(last))
def main():
 atomic(DEST/'STARTED.json',{'utc':stamp(),'pid':os.getpid(),'workers':3,'total_bytes':sum(a['bytes'] for a in ASSETS),'source':'official NSD S3 direct, requests trust_env=False','scope':'download only; no GPU or regression'})
 # Complete small metadata first, then up to three session downloads.
 ordered=sorted(ASSETS,key=lambda a:a['bytes'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,ordered))
 import numpy as np,pandas as pd,scipy.io
 behavior=pd.read_csv(DEST/'nsddata/ppdata/subj01/behav/responses.tsv',sep='\t')
 actual=np.concatenate([behavior.loc[behavior.SESSION==i,'73KID'].to_numpy(dtype=np.int64)-1 for i in range(1,38)])
 exp=Path('/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd/experiments/nsd/nsd_expdesign.mat')
 d=scipy.io.loadmat(exp,squeeze_me=True);ordering=np.asarray(d['masterordering']).reshape(-1)[:27750].astype(int);expected=np.asarray(d['subjectim'])[0,ordering-1].astype(int)-1
 if not np.array_equal(actual,expected):raise RuntimeError('Behavior and expdesign trial IDs differ')
 shared=set((np.asarray(d['sharedix']).reshape(-1)-1).astype(int).tolist());test=[int(i) for i in dict.fromkeys(actual.tolist()) if i in shared]
 plan=json.loads((ROOT/'regression-data-plan.json').read_text())
 if test!=plan['test_global_ids_in_native_order']:raise RuntimeError('Native test order differs')
 verify={'utc':stamp(),'n_trials':len(actual),'session_counts':{str(i):int((behavior.SESSION==i).sum()) for i in range(1,38)},'behavior_expdesign_exact':True,'native_test_order_exact':True,'n_test':len(test),'expdesign_sha256':digest(exp),'behavior_sha256':digest(DEST/'nsddata/ppdata/subj01/behav/responses.tsv')}
 atomic(DEST/'behavior_order_verification.json',verify)
 atomic(DEST/'DOWNLOAD_COMPLETE.json',{'utc':stamp(),'exit_status':0,'files':results,'behavior_verification':verify})
 print(json.dumps({'event':'download_complete','utc':stamp(),'n_files':len(results),'behavior_verified':True}),flush=True)
if __name__=='__main__':main()
