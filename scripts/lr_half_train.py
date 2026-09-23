"""Run only the authorized L training using the existing launcher and trainer."""
import fcntl,json,sys
from pathlib import Path
import stage3 as common
ROOT=common.ROOT
if __name__=='__main__':
    lock=(ROOT/'runtime/lr-half.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
    assert not run.exists(), 'Branch exists: inspect last COMPLETE before recovery; never replay source100'
    def event(**kw):
        with (ROOT/'logs/lr-half-events.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
        print(json.dumps(kw),flush=True)
    common.event=event
    common.execute('lr-half-train-to-110',[sys.executable,'scripts/run_train.py','--branch-lr-half','--target','110','--resume',str(ROOT/'runs/subject01-author-fp32-seed42/checkpoint-100')])
    state=json.loads((run/'checkpoint-110/progress.json').read_text())
    assert state['completed_epochs']==110 and state['global_optimizer_updates']==58520 and state['current_lr']==[5e-5]
    event(action='training_complete',epoch=110,updates=58520,exit_code=0)
