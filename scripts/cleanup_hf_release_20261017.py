"""Delete only explicit redundant download caches and superseded recovery states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

R=Path('/data1/matengyu/geyugong/neuroadapter-author-first').resolve()
B=R/'benchmarks/subj01-v1'
O=R/'reports/hf-release-20261017'
O.mkdir(exist_ok=True)
ap=argparse.ArgumentParser()
ap.add_argument('--phase',choices=['caches','old-recovery'],required=True)
ap.add_argument('--execute',action='store_true')
a=ap.parse_args()


def emit(x): print(json.dumps(x),flush=True)


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def checkpath(p):
 p=p.absolute()
 p.resolve().relative_to(R)
 assert not p.is_symlink(),str(p)
 for parent in p.parents:
  if parent==R:break
  assert not parent.is_symlink(),str(parent)
 return p


targets=[]
verified=[]
if a.phase=='caches':
 # Rehash the assembled files against their original public-source manifests.
 lock=json.loads((B/'mindeye-download-lock.json').read_text())
 for f in lock['files']:
  if not f.get('lfs_sha256'):continue
  p=B/'assets/mindeye'/f['path']
  assert p.is_file() and p.stat().st_size==f['size'] and sha(p)==f['lfs_sha256']
  verified.append(dict(path=str(p.relative_to(R)),sha256=f['lfs_sha256']))
  emit(dict(event='original_download_verified',path=f['path']))
 vd=json.loads((B/'vd-download-lock.json').read_text())
 for f in vd['files']:
  part=B/'assets/range-parts/vd'/Path(f['path']).parent.name/Path(f['path']).name
  if not part.exists():continue
  p=B/'assets/versatile-diffusion'/f['path']
  assert f.get('sha256') and p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
  verified.append(dict(path=str(p.relative_to(R)),sha256=f['sha256']))
  emit(dict(event='original_download_verified',path=f['path']))
 targets=[B/'assets/range-parts', B/'cache/mindbridge-pip']
 # Keep HF metadata/locks. Only obsolete partial payloads are discarded.
 targets += sorted((B/'assets/mindeye/.cache/huggingface').rglob('*.incomplete'))
else:
 final=O/'upload-result.json'
 if final.exists():
  upload=json.loads(final.read_text())
  assert upload['all_public_and_sha_verified']
  receipts=upload['repositories']
 else:
  receipts=json.loads((O/'upload-result.partial.json').read_text())
 mb=next(x for x in receipts if x['repo']=='gugabobo/mindbridge-subject01-reproduction')
 assert mb['sha_verified'] and mb['public']
 manifest=json.loads((O/'upload-manifest.json').read_text())
 pkg=next(x for x in manifest['packages'] if x['repo']==mb['repo'])
 for f in pkg['files']:
  p=R/f['source'];assert p.stat().st_size==f['size'] and sha(p)==f['sha256']
  verified.append(dict(path=f['source'],sha256=f['sha256']))
 complete=json.loads((B/'mindbridge-training-complete.json').read_text())
 assert complete['completed_epochs']==600 and complete['resume_exit_code']==0
 # Verify all final full-state files against the original completion audit.
 for f in complete['files']:
  p=R/f['path']
  assert p.is_file() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
  verified.append(dict(path=f['path'],sha256=f['sha256']))
 targets=[B/'processed/mindbridge-recovery-epoch420',
  B/'upstream/mindbridge/train_logs/benchmark-subj01-vanilla-seed42/best-interrupted-epoch420']

tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=R).decode().split('\0'))
files=[]
for p in targets:
 checkpath(p)
 if not p.exists():continue
 entries=[p] if p.is_file() else sorted(p.rglob('*'))
 for f in entries:
  checkpath(f)
  if not f.is_file():continue
  rel=str(f.relative_to(R));assert rel not in tracked,rel
  s=f.stat()
  files.append(dict(path=rel,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns))
# Do not race an active download, training, installation, or recovery worker.
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
blocked=[]
for line in processes.splitlines():
 if 'cleanup_remote' in line or 'cleanup_hf_release' in line:continue
 if any(t in line for t in ('download_assets_ranged.py','download_weights_ranged.py','mindbridge_resume_train.py','mindbridge_author_train.py','pip install')):
  blocked.append(line)
assert not blocked,'Relevant jobs are still running'
plan=dict(phase=a.phase,unix_time=time.time(),files=files,verified_sources=verified,
  targets=[str(x.relative_to(R)) for x in targets],allocated_bytes=sum(x['allocated'] for x in files),
  free_before_bytes=shutil.disk_usage(R).free,execute=a.execute)
(O/f'cleanup-{a.phase}-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
emit(dict(event='cleanup_plan',phase=a.phase,files=len(files),gib=plan['allocated_bytes']/1024**3,execute=a.execute))
if not a.execute:raise SystemExit(0)
# Recheck metadata and path boundary immediately before any removal.
for f in files:
 p=checkpath(R/f['path']);s=p.stat()
 assert s.st_size==f['size'] and s.st_mtime_ns==f['mtime_ns']
for p in targets:
 checkpath(p)
 if not p.exists():continue
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
result=dict(phase=a.phase,completed=True,unix_time=time.time(),deleted_files=len(files),
  freed_allocated_bytes=plan['allocated_bytes'],free_before_bytes=plan['free_before_bytes'],
  free_after_bytes=shutil.disk_usage(R).free,verified_source_count=len(verified),
  retained='All final inference weights, final complete resume states, raw data, metrics, generated outputs, and tracked files.')
(O/f'cleanup-{a.phase}-result.json').write_text(json.dumps(result,indent=2)+'\n')
emit(result)
