from pathlib import Path
import concurrent.futures,hashlib,json,os,time,requests,shutil
B=Path(__file__).resolve().parent;l=json.loads((B/'mindeye-download-lock.json').read_text())
for f in l['files']:
 if not f['path'].startswith('mindeye_models/'):continue
 dst=B/'assets/mindeye'/f['path'];dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():
  assert dst.stat().st_size==f['size'];continue
 cache=B/'assets/mindeye/.cache/huggingface/download'/Path(f['path']).parent
 prefixes=list(cache.glob('*.'+f['lfs_sha256']+'.incomplete'))
 prefix=prefixes[0] if prefixes else None;offset=prefix.stat().st_size if prefix else 0
 parts=B/'assets/range-parts'/dst.parent.name;parts.mkdir(parents=True,exist_ok=True)
 url=f"https://huggingface.co/datasets/{l['repo']}/resolve/{l['revision']}/{f['path']}"
 size=128*1024*1024
 ranges=[(n,start,min(start+size,f['size'])-1) for n,start in enumerate(range(offset,f['size'],size))]
 def fetch(item):
  n,start,end=item;p=parts/f'{start}-{end}.part'
  if p.exists() and p.stat().st_size==end-start+1:return p
  for attempt in range(4):
   try:
    with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,120)) as response:
     assert response.status_code==206,(response.status_code,start)
     assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),response.headers.get('Content-Range')
     with p.with_suffix('.tmp').open('wb') as stream:
      for chunk in response.iter_content(1024*1024):stream.write(chunk)
    assert p.with_suffix('.tmp').stat().st_size==end-start+1
    p.with_suffix('.tmp').replace(p);print('PART',dst.name,start,end,flush=True);return p
   except Exception as exc:
    print('RETRY',start,type(exc).__name__,str(exc)[:100],flush=True)
    if attempt==3:raise
    time.sleep(2**attempt)
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
