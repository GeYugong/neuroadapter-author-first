"""Inspect or finalize only this task's locked asset transfer files."""
import argparse
import hashlib
import json
from pathlib import Path
import time

R = Path('/data1/matengyu/geyugong/neuroadapter-author-first')
O = R / 'reports/mindeye-subject02-20261019'
D = R / 'data/mindeye-subject02'


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


p=argparse.ArgumentParser();p.add_argument('mode',choices=['inspect','finalize']);p.add_argument('--index',type=int,required=True);args=p.parse_args()
row=json.loads((O/'asset-plan.json').read_text())['files'][args.index]
target=D/row['path'];assert target.resolve().is_relative_to(D.resolve()) and not target.is_symlink()
target.parent.mkdir(parents=True,exist_ok=True)
receipt=O/'downloads'/f'{args.index:02d}.json';part=target.with_name(target.name+'.part')
if target.exists():
    known=json.loads(receipt.read_text());assert target.stat().st_size==row['bytes'] and digest(target)==known['actual_sha256']
    print(json.dumps({'state':'verified','index':args.index,'sha256':known['actual_sha256']}))
elif args.mode=='inspect':
    print(json.dumps({'state':'partial','index':args.index,'bytes':part.stat().st_size if part.exists() else 0,'path':str(part)}))
else:
    assert part.stat().st_size==row['bytes'];sha=digest(part)
    if row['sha256']:assert sha==row['sha256']
    receipt.parent.mkdir(exist_ok=True)
    value=dict(row,actual_sha256=sha,index=args.index,verified=True,time=time.time(),destination=str(target.relative_to(R)),transport='bounded local spool then scp; original partial prefix reused')
    receipt.write_text(json.dumps(value,indent=2)+'\n');part.replace(target)
    print(json.dumps({'state':'verified','index':args.index,'sha256':sha}))
