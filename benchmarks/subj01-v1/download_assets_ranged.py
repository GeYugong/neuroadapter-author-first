from pathlib import Path
import concurrent.futures,hashlib,json,os,time,requests,shutil
import argparse
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['test','vd'],required=True);args=p.parse_args()
B=Path(__file__).resolve().parent;l=json.loads((B/('mindeye-download-lock.json' if args.kind=='test' else 'vd-download-lock.json')).read_text())
if args.kind=='vd':l['files']=[dict(f,size=f['bytes'],lfs_sha256=f['sha256']) for f in l['files']]
for f in l['files']:
 if args.kind=='test' and '/test/test_subj01_' not in f['path']:continue
 if args.kind=='vd' and (f['path'].startswith(('pretrained_pth/','assets/')) or not f['lfs_sha256']):continue
 dst=B/('assets/mindeye' if args.kind=='test' else 'assets/versatile-diffusion')/f['path'];dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():
  assert dst.stat().st_size==f['size'];continue
 cache=B/('assets/mindeye/.cache/huggingface/download' if args.kind=='test' else 'assets/versatile-diffusion/.cache/huggingface/download')/Path(f['path']).parent
 prefixes=list(cache.glob('*.'+f['lfs_sha256']+'.incomplete'))
 prefix=prefixes[0] if prefixes else None;offset=prefix.stat().st_size if prefix else 0
 parts=B/'assets/range-parts'/args.kind/dst.parent.name/dst.name;parts.mkdir(parents=True,exist_ok=True)
 url="https://huggingface.co/"+("datasets/" if args.kind=="test" else "")+l["repo"]+"/resolve/"+l["revision"]+"/"+f["path"]
 with requests.get(url,proxies={'http':os.environ['HTTP_PROXY'],'https':os.environ['HTTPS_PROXY']},headers={'Range':'bytes=0-0'},stream=True,timeout=(20,90)) as resolved:
  resolved.raise_for_status();url=resolved.url
 direct=requests.Session();direct.trust_env=False
 size=128*1024*1024
 ranges=[(n,start,min(start+size,f['size'])-1) for n,start in enumerate(range(offset,f['size'],size))]
 def fetch(item):
  n,start,end=item;p=parts/f'{start}-{end}.part'
  if p.exists() and p.stat().st_size==end-start+1:return p
  tmp=p.with_suffix('.tmp')
  failures=0
  while (tmp.stat().st_size if tmp.exists() else 0)<end-start+1:
   position=start+(tmp.stat().st_size if tmp.exists() else 0)
   last=min(end,position+8*1024*1024-1)
   try:
    with direct.get(url,headers={'Range':f'bytes={position}-{last}'},stream=True,timeout=(20,90)) as response:
     assert response.status_code==206,(response.status_code,position)
     assert response.headers['Content-Range'].startswith(f'bytes {position}-{last}/'),response.headers.get('Content-Range')
     with tmp.open('ab') as stream:
      for chunk in response.iter_content(1024*1024):stream.write(chunk)
    failures=0
   except Exception as exc:
    failures+=1;print('RESUME_RETRY',position,type(exc).__name__,flush=True)
    if failures>=8:raise
    time.sleep(min(2**failures,20))
  assert tmp.stat().st_size==end-start+1
  tmp.replace(p);print('PART',dst.name,start,end,flush=True);return p
 print('RANGED_RESUME',f['path'],'prefix',offset,'remaining',f['size']-offset,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:files=list(pool.map(fetch,ranges))
 tmp=dst.with_suffix('.assembling')
 with tmp.open('wb') as out:
  if prefix:
   with prefix.open('rb') as src:shutil.copyfileobj(src,out,8*1024*1024)
  for part in files:
   with part.open('rb') as src:shutil.copyfileobj(src,out,8*1024*1024)
 h=hashlib.sha256()
 with tmp.open('rb') as stream:
  for c in iter(lambda:stream.read(8388608),b''):h.update(c)
 assert h.hexdigest()==f['lfs_sha256'] and tmp.stat().st_size==f['size']
 tmp.replace(dst)
 (B/'assets'/('verified-'+dst.name+'.json')).write_text(json.dumps(dict(f,sha256=h.hexdigest(),verified=True),indent=2))
 print('VERIFIED',f['path'],h.hexdigest(),flush=True)
