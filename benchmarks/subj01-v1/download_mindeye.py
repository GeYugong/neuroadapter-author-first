from pathlib import Path
import json,hashlib,time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
B=Path(__file__).resolve().parent
lock=json.loads((B/'mindeye-download-lock.json').read_text())
files=[f for f in lock['files'] if 'metadata' not in f['path'] or f['path']=='webdataset_avg_split/metadata_subj01.json']
def download(f):
    print('START',f['path'],flush=True)
    p=Path(hf_hub_download(lock['repo'],f['path'],repo_type='dataset',revision=lock['revision'],local_dir=B/'assets/mindeye',token=False))
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    digest=h.hexdigest()
    assert p.stat().st_size==f['size']
    if f['lfs_sha256']:assert digest==f['lfs_sha256']
    result=dict(f,sha256=digest,verified=True)
    (B/'assets'/('verified-'+p.name+'.json')).write_text(json.dumps(result,indent=2))
    print('VERIFIED',f['path'],digest,flush=True)
    return result
with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(download,files))
(B/'mindeye-download-complete.json').write_text(json.dumps(results,indent=2))
