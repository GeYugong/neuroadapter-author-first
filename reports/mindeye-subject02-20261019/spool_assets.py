"""Resume official assets through bounded local files and short SCP sessions."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import requests

H=Path(__file__).resolve().parent;P=json.loads((H/'asset-plan.json').read_text());S=H/'asset-spool';S.mkdir(exist_ok=True)
R='/data1/matengyu/geyugong/neuroadapter-author-first'
OPTIONS=['-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=20']


def remote(mode,index):
    cmd=['ssh',*OPTIONS,'neuroadapter-4090',f'cd {R} && runtime/env/bin/python -B scripts/mindeye_asset_finalize.py {mode} --index {index}']
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    if r.returncode:raise RuntimeError('Scoped remote verification failed')
    return json.loads(r.stdout)


def send(index):
    item=P['files'][index];state=remote('inspect',index)
    if state['state']=='verified':return state
    target=S/f'{index:02d}.part';remote_part=state['path']
    if not target.exists() and state['bytes']:
        r=subprocess.run(['scp',*OPTIONS,f'neuroadapter-4090:{remote_part}',str(target)],capture_output=True,timeout=240)
        if r.returncode:raise RuntimeError('Partial prefix SCP failed')
    for attempt in range(3):
        offset=target.stat().st_size if target.exists() else 0
        if offset==item['bytes']:break
        assert offset<item['bytes']
        try:
            with requests.get(item['url'],headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90)) as resp:
                resp.raise_for_status()
                assert resp.status_code==(206 if offset else 200)
                if offset:assert resp.headers['Content-Range'].startswith(f'bytes {offset}-')
                if item.get('etag'):assert resp.headers.get('ETag')==item['etag']
                with target.open('ab') as f:
                    for chunk in resp.iter_content(4*1024**2):f.write(chunk)
            break
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(3)
    assert target.stat().st_size==item['bytes']
    with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if item['sha256']:assert sha==item['sha256']
    r=subprocess.run(['scp',*OPTIONS,str(target),f'neuroadapter-4090:{remote_part}'],capture_output=True,timeout=300)
    if r.returncode:raise RuntimeError('Final asset SCP failed')
    result=remote('finalize',index);assert result['sha256']==sha
    assert target.resolve().parent==S.resolve() and not target.is_symlink()
    target.unlink()
    return result


if __name__=='__main__':
    done=[];failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(send,i):i for i in range(len(P['files']))}
        for f in concurrent.futures.as_completed(futures):
            i=futures[f]
            try:
                result=f.result();done.append(result)
                print(json.dumps({'event':'asset_verified','completed':len(done),**result}),flush=True)
            except Exception as e:
                failures.append({'index':i,'error_type':type(e).__name__})
                print(json.dumps({'event':'asset_failed',**failures[-1]}),flush=True)
            (H/'spool-status.json').write_text(json.dumps({'state':'complete' if len(done)==len(P['files']) else 'preparing','completed':len(done),'total':len(P['files']),'failures':failures,'results':done,'time':time.time()},indent=2)+'\n')
    if failures:raise SystemExit(1)
