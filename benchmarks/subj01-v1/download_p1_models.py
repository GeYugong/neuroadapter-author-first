from pathlib import Path
import concurrent.futures,datetime,hashlib,json,os,time,requests
B=Path('/data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1');A=B/'assets/p1-models';LOCK=B/'p1-model-download-lock.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');os.replace(q,p)
def fetch(f):
 dst=A/f['destination'];dst.parent.mkdir(parents=True,exist_ok=True);part=dst.with_suffix(dst.suffix+'.part');audit=dst.with_suffix(dst.suffix+'.verified.json')
 def verify(p):
  assert p.stat().st_size==f['bytes'];h=sha(p)
  if f['sha256']:assert h==f['sha256'],(f['name'],'SHA mismatch')
  return h
 if dst.exists():
  h=verify(dst);assert audit.exists() and json.loads(audit.read_text())['actual_sha256']==h;return json.loads(audit.read_text())
 direct=requests.Session();direct.trust_env=False;url=f['url'];etag=None
 if 'huggingface.co/' in url:
  resolver=requests.Session();resolver.trust_env=False
  with resolver.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=(20,90),proxies={'https':'http://127.0.0.1:19876'}) as r:r.raise_for_status();url=r.url;etag=r.headers.get('ETag')
 else:
  h=direct.head(url,timeout=40);h.raise_for_status();etag=h.headers.get('ETag');assert etag==f['etag']
 failures=0
 while (part.stat().st_size if part.exists() else 0)<f['bytes']:
  offset=part.stat().st_size if part.exists() else 0;end=min(offset+8*1024*1024,f['bytes'])-1
  try:
   headers={'Range':f'bytes={offset}-{end}'}
   if f.get('etag'):headers['If-Match']=f['etag']
   with direct.get(url,headers=headers,stream=True,timeout=(20,90)) as r:
    r.raise_for_status();assert r.status_code==206;assert r.headers.get('Content-Range','').startswith(f'bytes {offset}-{end}/')
    with part.open('ab') as out:
     for c in r.iter_content(1024*1024):
      if c:out.write(c)
   failures=0
   if offset//(256*1024*1024)!=part.stat().st_size//(256*1024*1024):print(json.dumps({'event':'progress','name':f['name'],'bytes':part.stat().st_size,'utc':now()}),flush=True)
  except Exception as e:
   failures+=1;print(json.dumps({'event':'retry','name':f['name'],'attempt':failures,'error':str(e),'utc':now()}),flush=True)
   if failures>=8:raise
   time.sleep(min(2**failures,20))
 h=verify(part);os.replace(part,dst);result=dict(f,actual_sha256=h,actual_etag=etag,verified_utc=now(),path=str(dst));save(audit,result);print(json.dumps({'event':'verified','name':f['name'],'sha256':h,'utc':now()}),flush=True);return result
if __name__=='__main__':
 A.mkdir(parents=True,exist_ok=True);lock=json.loads(LOCK.read_text());save(A/'STARTED.json',{'pid':os.getpid(),'utc':now(),'workers':3,'total_bytes':lock['total_bytes'],'lock_sha256':sha(LOCK)})
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:results=list(p.map(fetch,sorted(lock['files'],key=lambda f:f['bytes'])))
 save(A/'DOWNLOAD_COMPLETE.json',{'utc':now(),'exit_status':0,'files':results});print(json.dumps({'event':'complete','utc':now()}),flush=True)
