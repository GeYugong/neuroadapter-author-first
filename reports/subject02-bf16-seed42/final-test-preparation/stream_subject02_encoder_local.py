"""Local public-HF to authorized-project SSH stream; no local weight/cache copy."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time
import traceback
import requests

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'encoder-download-plan.json').read_text())
SSH=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','neuroadapter-4090']
REMOTE='/data1/matengyu/geyugong/neuroadapter-author-first'


def send(index):
    item=PLAN['files'][index]
    url=f'https://huggingface.co/{PLAN["repository"]}/resolve/{PLAN["revision"]}/{item["path"]}'
    for attempt in range(1,4):
        child=subprocess.Popen(SSH+[f'cd {REMOTE} && runtime/env/bin/python -B scripts/receive_subject02_encoder.py --index {index}'],
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            line=child.stdout.readline()
            if not line:raise RuntimeError('Receiver did not initialize')
            ready=json.loads(line)
            if ready['state']=='already_verified':
                child.stdin.close();assert child.wait(timeout=30)==0
                return dict(index=index,sha256=item['sha256'],already_verified=True)
            assert ready['state']=='ready'
            offset=ready['offset']
            session=requests.Session()
            headers={'Range':f'bytes={offset}-'} if offset else {}
            if offset<item['bytes']:
                with session.get(url,headers=headers,stream=True,timeout=(20,90)) as response:
                    response.raise_for_status()
                    assert response.status_code==(206 if offset else 200)
                    if offset:assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                    for block in response.iter_content(4*1024**2):
                        child.stdin.write(block)
            child.stdin.close()
            result=child.stdout.read();error=child.stderr.read()
            assert child.wait(timeout=120)==0,'Receiver rejected stream: '+error.decode(errors='replace')[-500:]
            report=json.loads(result.decode().splitlines()[-1]);assert report['state']=='verified'
            return report
        except Exception as error:
            try:child.stdin.close()
            except Exception:pass
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:raise RuntimeError('Receiver did not exit; inspect before resuming')
            print(json.dumps({'index':index,'attempt':attempt,'error_type':type(error).__name__}),flush=True)
            if attempt==3:raise
            time.sleep(5*attempt)


def main():
    state=HERE/'encoder-stream-status.json'
    assert not state.exists(),'Inspect prior stream before retry'
    state.write_text(json.dumps({'state':'running','time':time.time(),'total':32})+'\n')
    done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(send,i) for i in range(len(PLAN['files']))]):
            done.append(future.result())
            value={'state':'streaming','completed':len(done),'total':32,'time':time.time(),'files':done}
            state.write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps({'completed':len(done),'total':32}),flush=True)
    state.write_text(json.dumps({'state':'complete','time':time.time(),'files':done},indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception:
        (HERE/'encoder-stream-failure.txt').write_text(traceback.format_exc())
        raise
