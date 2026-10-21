"""Relay bounded task files between the two authorized project roots."""
import argparse,json,pathlib,subprocess,time
H=pathlib.Path(__file__).resolve().parent
S='/data/matengyu/geyugong/neuroadapter-author-first'
D='/data1/matengyu/geyugong/neuroadapter-author-first'
source=['ssh','-i','C:/Users/gyg17/.ssh/id_ed25519_neuroadapter_5090','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15','10.8.176.68']
destination=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15','neuroadapter-4090']
p=argparse.ArgumentParser();p.add_argument('mode',choices=['environment','assets','weights']);args=p.parse_args()
def relay(src,dst,label):
    a=subprocess.Popen(source+[src],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    b=subprocess.Popen(destination+[dst],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    total=0;t=time.time()
    try:
        while chunk:=a.stdout.read(4*2**20):b.stdin.write(chunk);total+=len(chunk)
        b.stdin.close()
        e1=a.stderr.read().decode();c1=a.wait()
        out=b.stdout.read().decode();e2=b.stderr.read().decode();c2=b.wait()
        assert c1==c2==0,(label,c1,c2,e1,e2)
        print('RELAYED',label,total,round(time.time()-t,1),out.strip(),flush=True)
    finally:
        for proc in [a,b]:
            if proc.poll() is None:proc.terminate()

if args.mode=='environment':
    relay(f'cd {S} && tar -C runtime/umbrae-env/lib/python3.11/site-packages -cf - .',
          f'cd {D} && test ! -e runtime/umbrae-site && mkdir runtime/umbrae-site && tar -C runtime/umbrae-site -xf -','environment')
elif args.mode=='assets':
    plan=json.loads((H/'asset-plan.json').read_text())
    rows=[r for r in plan['files'] if 'local' not in r and 'model-000' not in r['path'] and '/nsd-test/' not in r['path']]
    for row in rows:
        name=row['path'];parent=str(pathlib.PurePosixPath(name).parent)
        existing=subprocess.run(destination+[f'cd {D} && test -f {name} && sha256sum {name}'],capture_output=True,text=True,encoding='utf-8')
        if existing.returncode==0:
            assert existing.stdout.split()[0]==row['sha256']
            print('ALREADY_VERIFIED',name,flush=True);continue
        assert subprocess.run(source+[f'cd {S} && test -f {name}']).returncode==0,'Source asset is not complete: '+name
        # Source assets are immutable; receivers validate every digest independently.
        relay(f'cd {S} && test -f {name} && cat {name}',
              f'cd {D} && runtime/env/bin/python -B scripts/receive_umbrae_4090_asset.py --path {name}',name)
    relay(f'cd {S} && tar -C runtime/umbrae/assets/evaluation-extras -cf - .',
          f'cd {D} && mkdir -p runtime/umbrae/assets/evaluation-extras && tar -C runtime/umbrae/assets/evaluation-extras -xf -','evaluation-extras')
else:
    plan=json.loads((H/'asset-plan.json').read_text())
    for row in [r for r in plan['files'] if 'model-000' in r['path']]:
        name=row['path']
        for attempt in range(180):
            status=subprocess.run(destination+[f'cd {D} && cat reports/umbrae-subject01-20261021-4090/stream-status.json'],capture_output=True,text=True,encoding='utf-8')
            if status.returncode==0:
                state=json.loads(status.stdout)
                if state.get('phase')=='waiting_for_shard' and state.get('path')==name:break
            time.sleep(5)
        else:raise RuntimeError('Weight receiver did not request the expected shard: '+name)
        relay(f'cd {S} && test -f {name} && cat {name}',
              f'cd {D} && test -p reports/umbrae-subject01-20261021-4090/model-weights.fifo && cat > reports/umbrae-subject01-20261021-4090/model-weights.fifo',name)
