"""One launch of the audited independent MindEye Subject2 training branches."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from mindeye_subject02_common import R,O,D,S,COMPAT,sha,write


def main():
    lock=(O/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'controller-started.json').exists(),'Never duplicate an existing launch'
    assert json.loads((O/'data-ready.json').read_text())['complete']
    assert json.loads((O/'loader-audit.json').read_text())['passed']
    pre={k:json.loads((O/f'preflight-{k}/audit.json').read_text()) for k in ['high','low']}
    assert all(v['passed'] and v['resumed_model_optimizer_RNG_scaler_scheduler_exact'] for v in pre.values())
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPUs occupied; no launch'
    assert all(not (R/f'runs/mindeye-subject02-{k}').exists() for k in pre)
    # Three full states at atomic-save peak plus one inference-best per branch.
    expected_peak=sum(v['checkpoint_bytes']*(3+1/3) for v in pre.values())
    assert shutil.disk_usage(R).free>expected_peak+25*1024**3,'Insufficient projected retention reserve'
    code=['scripts/mindeye_subject02_common.py','scripts/train_mindeye_subject02.py','scripts/prepare_mindeye_subject02.py','scripts/launch_mindeye_subject02.py']
    code += [str(p.relative_to(R)) for p in sorted(COMPAT.rglob('*.py'))]
    code += [str((S/name).relative_to(R)) for name in ['models.py','utils.py','Train_MindEye.py','train_autoencoder.py','convnext.py']]
    assets=[D/x['path'] for x in json.loads((O/'asset-plan.json').read_text())['files']]
    frozen=dict(independent_validation_authorized=True,train=8559,validation=300,test_in_training=False,
        code_sha256={name:sha(R/name) for name in code},
        asset_stats=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in assets],
        source_commit=subprocess.check_output(['git','-C',str(S.parent),'rev-parse','HEAD'],text=True).strip(),
        configurations={k:v['config'] for k,v in pre.items()},
        checkpoint_retention='Latest2 complete continuation states and best validation inference state per new branch; final fixed endpoints preserved',
        projected_checkpoint_peak_bytes=int(expected_peak),minimum_disk_reserve_gib=25,
        no_automatic_restart=True,time=time.time())
    assert frozen['source_commit']=='6b30ea53c0f98b0a79b8f6828b6787a5913fd08d'
    write(O/'frozen-plan.json',frozen)
    write(O/'controller-started.json',dict(pid=os.getpid(),time=time.time(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    children=[]
    for kind,gpu in [('high',0),('low',1)]:
        log=(O/f'{kind}-train.log').open('x')
        command=[str(R/'runtime/env/bin/python'),'-B',str(R/'scripts/train_mindeye_subject02.py'),'train','--kind',kind]
        child=subprocess.Popen(command,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
        children.append((kind,child,log))
        write(O/f'{kind}-launched.json',dict(pid=child.pid,gpu=gpu,command=command,time=time.time()))
    recorded=set()
    while True:
        for kind,child,log in children:
            if child.poll() is not None and kind not in recorded:
                log.close();write(O/f'{kind}-train-exit.json',dict(returncode=child.returncode,pid=child.pid,time=time.time()));recorded.add(kind)
        status=dict(phase='running' if len(recorded)<2 else ('training_complete' if all(c.returncode==0 for _,c,_ in children) else 'failed_needs_inspection'),
            children=[dict(kind=k,pid=c.pid,returncode=c.poll()) for k,c,_ in children],time=time.time())
        write(O/'controller-status.json',status,True)
        if len(recorded)==2:break
        time.sleep(20)


if __name__=='__main__':
    try:main()
    except Exception:
        write(O/f'controller-failure-{int(time.time())}.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
