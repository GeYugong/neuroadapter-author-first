"""Authorized cumulative 90/100 orchestration; reuse existing execution and reports."""
import argparse
import fcntl
import json
import os
import sys
import time
import stage3 as common
ROOT,RUN=common.ROOT,common.RUN

def event(**fields):
    record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**fields)
    with (ROOT/'logs/stage5-events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',type=int,choices=[90,100],required=True)
    target=parser.parse_args().target
    lock=(ROOT/'runtime/stage5.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (ROOT/'reports/STAGE4_COMPLETE').exists()
    previous=target-10;checkpoint=RUN/f'checkpoint-{previous}'
    assert (checkpoint/'COMPLETE').exists() and (ROOT/f'reports/epoch-{previous}/COMPLETE').exists()
    state=json.loads((checkpoint/'progress.json').read_text())
    assert state['completed_epochs']==previous and state['global_optimizer_updates']==previous*532
    if any(int(p.name.split('-')[1])>previous for p in RUN.glob('checkpoint-*') if p.name.split('-')[1].isdigit()):
        raise RuntimeError('Later checkpoint exists; inspect before recovery')
    if (ROOT/f'reports/epoch-{target}').exists():raise FileExistsError('Target report exists')
    common.event=event
    event(action='start',pid=os.getpid(),target=target,max_cumulative_epoch=100,source=str(checkpoint))
    common.execute(f'train-to-{target}',[sys.executable,'scripts/run_train.py','--stage5','--target',str(target),'--resume',str(checkpoint)])
    state=json.loads((RUN/f'checkpoint-{target}/progress.json').read_text())
    assert state['completed_epochs']==target and state['global_optimizer_updates']==target*532
    common.execute(f'eval-{target}',[sys.executable,'scripts/evaluate_stage.py','--run',str(RUN),'--epoch',str(target)])
    common.summarize(target,milestones=(5,10,20,30,40,50,60,70,80,90,100),stop_epoch=100)
    event(action='stage_complete',epoch=target,updates=state['global_optimizer_updates'])
    if target==100:(ROOT/'reports/STAGE5_COMPLETE').write_text('Cumulative epoch 100 completed; stop and await user.\n')
    event(action='completed_stopped',epoch=target,exit_code=0)

if __name__=='__main__':
    try:main()
    except Exception as e:
        event(action='failed',error=repr(e),exit_code=1)
        raise
