"""Receive a declared UMBRAE asset while retaining the disk reserve."""
import argparse,hashlib,json,os,pathlib,shutil,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'reports/umbrae-subject01-20261021-4090'
p=argparse.ArgumentParser();p.add_argument('--path',required=True);a=p.parse_args()
plan=json.loads((O/'asset-plan.json').read_text());row=next(r for r in plan['files'] if r['path']==a.path)
assert 'model-000' not in a.path and '/nsd-test/' not in a.path
target=R/a.path;assert target.resolve().is_relative_to((R/'runtime/umbrae').resolve())
target.parent.mkdir(parents=True,exist_ok=True)
assert not target.exists() and not target.is_symlink()
assert shutil.disk_usage(R).free>row['bytes']+25*2**30
part=target.with_name(target.name+'.part')
assert not part.exists() or part.stat().st_size==0,'Nonempty partial requires explicit range recovery'
h=hashlib.sha256();n=0
with part.open('ab') as f:
    while chunk:=sys.stdin.buffer.read(4*2**20):
        n+=len(chunk);assert n<=row['bytes'];f.write(chunk);h.update(chunk)
    f.flush();os.fsync(f.fileno())
assert n==row['bytes'] and h.hexdigest()==row['sha256']
part.replace(target)
print(json.dumps(dict(path=a.path,bytes=n,sha256=h.hexdigest(),verified=True)))
