"""Receive one pinned public Subject2 encoder file through a bounded stdin stream."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

R=Path('/data1/matengyu/geyugong/neuroadapter-author-first')
O=R/'reports/subject02-bf16-seed42/final-test-preparation'
DEST=R/'runtime/subject02-brain-encoder'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    plan=json.loads((O/'encoder-download-plan.json').read_text())
    assert plan['revision']=='d8a978abb212eb2965b5d01673f96536b77e2ea0'
    assert 0<=args.index<len(plan['files'])==32
    row=plan['files'][args.index]
    assert row['path'].startswith('dinov2_q_transformer/schaefer/subj_02/')
    target=DEST/row['path']
    assert target.resolve().is_relative_to(DEST.resolve()) and DEST.resolve()==DEST
    for p in target.parents:
        if p==R:break
        assert not p.is_symlink()
    target.parent.mkdir(parents=True,exist_ok=True)
    lock=target.with_name(target.name+'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not target.is_symlink()
    if target.exists():
        assert target.stat().st_size==row['bytes'] and digest(target)==row['sha256']
        print(json.dumps({'state':'already_verified','index':args.index}),flush=True)
        return
    part=target.with_name(target.name+'.part');assert not part.is_symlink()
    offset=part.stat().st_size if part.exists() else 0
    assert offset<=row['bytes']
    assert shutil.disk_usage(R).free>row['bytes']-offset+25*1024**3
    print(json.dumps({'state':'ready','index':args.index,'offset':offset}),flush=True)
    with part.open('ab') as f:
        received=offset
        while chunk:=sys.stdin.buffer.read(4*1024**2):
            assert received+len(chunk)<=row['bytes'],'Stream exceeds pinned length'
            f.write(chunk);received+=len(chunk)
        f.flush();os.fsync(f.fileno())
    assert part.stat().st_size==row['bytes'],'Partial retained; resume required'
    assert digest(part)==row['sha256'],'Checksum mismatch; preserve partial for inspection'
    part.replace(target)
    receipt=O/'encoder-downloads'/f'{args.index:02}.json';receipt.parent.mkdir(exist_ok=True)
    value=dict(row,index=args.index,verified=True,time=time.time(),destination=str(target.relative_to(R)))
    temp=receipt.with_name(receipt.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(receipt)
    print(json.dumps({'state':'verified','index':args.index,'sha256':row['sha256'],'bytes':row['bytes']}),flush=True)


if __name__=='__main__':main()
