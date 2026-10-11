"""One explicit disk-failure recovery of the first two best8 spooled samples."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,shutil,fcntl,traceback
R=Path(__file__).resolve().parents[1]
O=R/'reports/l450-best8-test982'
PY=R/'runtime/env/bin/python'
SOURCE=R/'scripts/evaluate_best8.py'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,d):
    tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def stamp(**kw):return dict(unix_time=time.time(),**kw)

def evidence():
    a=json.loads((O/'disk-recovery1-audit.json').read_text())
    assert sha(Path(__file__))==a['recovery_script_sha256']
    assert sha(SOURCE)==a['base_script_sha256']
    return a

def check_spools(a):
    import numpy as np
    assert not list((O/'records').glob('*.json')) and not list((O/'selected').glob('*.npz'))
    assert {p.name for p in (O/'queue').glob('*.npz')}==set(a['queue_sha256'])
    for name,h in a['queue_sha256'].items():assert sha(O/'queue'/name)==h

def run_child(mode):
    a=evidence()
    source=SOURCE.read_text()
    # Preserve the original STOP as evidence; use a recovery-specific sentinel.
    assert source.count("O/'STOP'")==4
    source=source.replace("O/'STOP'","O/'RECOVERY1_STOP'")
    needle="            i=int(batch['image_id'].item())"
    assert source.count(needle)==1
    source=source.replace(needle,needle+"\n            if not preflight and i in "+repr(set(a['resume_skip_image_ids']))+":\n                continue")
    for name in ['generation-status','selection-status']:
        source=source.replace(name,name+'-recovery1')
    context={'__name__':'best8_disk_recovery','__file__':str(SOURCE)}
    sys.path.insert(0,str(R))
    exec(compile(source,str(SOURCE),'exec'),context)
    context[mode]()

def controller():
    a=evidence()
    lock=(O/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'recovery1-started.json').exists()
    check_spools(a)
    write(O/'recovery1-started.json',stamp(pid=os.getpid(),command=sys.argv))
    status=O/'recovery1-status.json'
    children=[]
    try:
        while True:
            free=shutil.disk_usage(R).free
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if free>=4*1024**3 and not gpu:break
            write(status,stamp(status='waiting_for_disk_and_idle_gpus',free_bytes=free,
                required_free_bytes=4*1024**3,gpu_occupied=bool(gpu),
                preserved_samples=2,selected_samples=0))
            time.sleep(60)
        check_spools(a)
        for mode,gpu in [('select','1'),('generate','0')]:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            log=(O/f'{mode}-recovery1.log').open('x')
            cmd=[str(PY),'-B',str(Path(__file__).resolve()),mode]
            p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((mode,p,log))
            write(O/f'{mode}-recovery1-started.json',stamp(pid=p.pid,command=cmd))
        write(status,stamp(status='generating_and_selecting',resumed_queue_samples=2))
        seen=set()
        while True:
            failed=False
            for mode,p,f in children:
                rc=p.poll()
                if rc is not None and mode not in seen:
                    write(O/f'{mode}-recovery1-exit.json',stamp(returncode=rc));seen.add(mode)
                if rc not in (None,0):failed=True
            if failed:
                (O/'RECOVERY1_STOP').write_text('Inspect child logs; do not restart automatically.')
                for mode,p,f in children:
                    if p.poll() is None:p.wait(timeout=240)
                raise RuntimeError('Recovery child failed; preserve all results and investigate')
            if len(seen)==2:break
            time.sleep(10)
        for _,_,f in children:f.close()
        write(status,stamp(status='scoring'))
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
        with (O/'score-stage-recovery1.log').open('x') as f:
            rc=subprocess.run([str(PY),'-B',str(Path(__file__).resolve()),'score'],
                cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        write(O/'score-stage-recovery1-exit.json',stamp(returncode=rc))
        assert rc==0
        write(status,stamp(status='metrics_complete_review_pending'))
    except BaseException:
        write(status,stamp(status='failed',traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    assert len(sys.argv)==2
    if sys.argv[1]=='controller':controller()
    else:
        assert sys.argv[1] in ['generate','select','score']
        run_child(sys.argv[1])
