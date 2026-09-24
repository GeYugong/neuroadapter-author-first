"""Bounded seed43 training controller; original author loop, no intermediate selection."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import stage3 as common
ROOT=common.ROOT
RUN=ROOT/'runs/subject01-author-fp32-seed43-lrhalf-at100'
REPORT=ROOT/'reports/seed43-hl-replication'

def event(**kw):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    with (ROOT/'logs/seed43-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume');args=parser.parse_args()
    lock=(ROOT/'runtime/seed43-replication.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    common.event=event
    source=None
    if args.resume:
        source=Path(args.resume).resolve()
        assert source.parent==RUN and (source/'COMPLETE').exists()
        complete=list(RUN.glob('checkpoint-*/COMPLETE'))
        assert source==max((p.parent for p in complete),key=lambda p:int(p.name.split('-')[-1])), 'Only latest full state is eligible'
    else:
        assert not RUN.exists(), 'Inspect prior run, then explicitly resume latest COMPLETE'
    event(action='controller_start',pid=os.getpid(),resume=str(source) if source else None)
    try:
        for target in (100,120):
            if source and json.loads((source/'progress.json').read_text())['completed_epochs']>=target:continue
            cmd=[sys.executable,'scripts/run_train.py','--seed43-replication','--target',str(target)]
            if source:cmd+=['--resume',str(source)]
            common.execute(f'seed43-train-to-{target}',cmd)
            source=RUN/f'checkpoint-{target}'
            assert (source/'COMPLETE').exists()
            state=json.loads((source/'progress.json').read_text())
            assert (state['completed_epochs'],state['global_optimizer_updates'],state['skipped_updates'])==(target,target*532,0)
            assert state['current_lr']==[1e-4 if target==100 else 5e-5]
            assert (RUN/f'exports/epoch-{target}/inference.pt').exists()
            event(action='training_milestone',epoch=target,updates=state['global_optimizer_updates'])
        (REPORT/'TRAINING_COMPLETE').write_text('120 epochs / 63840 actual updates; independent evaluation pending.\n')
        event(action='training_controller_complete',exit_code=0,evaluation_pending=True)
    except Exception as exc:
        event(action='controller_failed',error=repr(exc));raise
if __name__=='__main__':main()
