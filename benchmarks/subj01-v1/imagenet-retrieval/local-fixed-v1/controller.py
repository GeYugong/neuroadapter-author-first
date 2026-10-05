"""Run authorized retrieval and independent delivery stages with real exit codes."""
import argparse
import os
import subprocess
import sys
from common import *


def main():
    p = argparse.ArgumentParser(); p.add_argument('--resume', action='store_true'); a = p.parse_args()
    prefix = 'imagenet-local-fixed-v1-' + time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    started = dict(pid=os.getpid(), utc=utc(), log_prefix=prefix, resume=a.resume, **limitations())
    write_json(HERE / f'controller-started-{prefix}.json', started)
    if not (HERE / 'controller-started.json').exists():
        write_json(HERE / 'controller-started.json', started)
    write_json(HERE / 'controller-current.json', started)
    for stage in ['scan', 'deliver']:
        log = ROOT / 'logs' / f'{prefix}-{stage}.log'
        command = [str(ROOT / 'runtime/env/bin/python'), '-B', str(HERE / f'{stage}.py')]
        if stage == 'scan' and a.resume:
            command.append('--resume')
        with log.open('xb') as f:
            result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        log.with_suffix('.exit').write_text(str(result.returncode) + '\n')
        write_json(HERE / f'{stage}-exit.json', dict(utc=utc(), exit_code=result.returncode,
            command=command, log=str(log), log_sha256=sha(log), **limitations()))
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
