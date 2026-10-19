"""Receive one locked official MindEye Subject 2 asset without changing old files."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

R = Path('/data1/matengyu/geyugong/neuroadapter-author-first')
O = R / 'reports/mindeye-subject02-20261019'
D = R / 'data/mindeye-subject02'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    index = parser.parse_args().index
    plan = json.loads((O/'asset-plan.json').read_text())
    assert plan['subject'] == 2 and len(plan['files']) == 21
    row = plan['files'][index]
    target = D / row['path']
    assert target.resolve().is_relative_to(D.resolve()) and D.resolve() == D
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.is_symlink()
    lock = target.with_name(target.name+'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = O / 'downloads' / f'{index:02d}.json'
    if target.exists():
        known = json.loads(receipt.read_text())
        assert target.stat().st_size == row['bytes'] and sha(target) == known['actual_sha256']
        print(json.dumps(dict(state='already_verified', index=index)), flush=True)
        return
    part = target.with_name(target.name+'.part')
    assert not part.is_symlink()
    offset = part.stat().st_size if part.exists() else 0
    assert 0 <= offset <= row['bytes']
    assert shutil.disk_usage(R).free > row['bytes'] - offset + 25*1024**3
    print(json.dumps(dict(state='ready', index=index, offset=offset)), flush=True)
    with part.open('ab') as f:
        received = offset
        while block := sys.stdin.buffer.read(4*1024**2):
            assert received+len(block) <= row['bytes']
            f.write(block)
            received += len(block)
        f.flush()
        os.fsync(f.fileno())
    assert part.stat().st_size == row['bytes'], 'Partial retained for bounded transport resume'
    digest = sha(part)
    if row['sha256'] is not None:
        assert digest == row['sha256'], 'Integrity failure; preserve partial for inspection'
    receipt.parent.mkdir(exist_ok=True)
    value = dict(row, actual_sha256=digest, index=index, verified=True, time=time.time(),
                 destination=str(target.relative_to(R)),
                 source_hash='Official HF LFS SHA256' if row['sha256'] else 'Computed from official HTTPS payload; size and ETag locked')
    receipt.write_text(json.dumps(value, indent=2)+'\n')
    part.replace(target)
    print(json.dumps(dict(state='verified', index=index, sha256=digest, bytes=row['bytes'])), flush=True)


if __name__ == '__main__':
    main()
