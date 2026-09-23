"""Bounded L110->120 controller; existing author training and evaluation functions."""
import argparse,fcntl,json,sys
from pathlib import Path
import stage3 as common
import lr_half_evaluate as old
import evaluate_stage as ev
ROOT=common.ROOT;RUN=old.L;REPORT=ROOT/'reports/lr-half-to120';CFG=ROOT/'configs'
def phase(name):
    if name=='generate':
        state=old.read(RUN/'checkpoint-120/progress.json')
        assert (state['completed_epochs'],state['global_optimizer_updates'],state['current_lr'])==(120,63840,[5e-5])
        assert not (RUN/'decoded/epoch-120').exists()
        ev.decode(RUN,120,'val',CFG/'val_ids.txt',RUN/'decoded/epoch-120/val')
        ev.decode(RUN,120,'train',CFG/'atlas_train16_ids.txt',RUN/'decoded/epoch-120/train')
    elif name=='score':
        old.identity(RUN,120,63840,RUN/'decoded/epoch-120/val')
        for i in old.ids('val_ids.txt'):old.verify_file(RUN/f'decoded/epoch-120/val/image_{i:06d}.npz',i,RUN/f'decoded/epoch-110/val/image_{i:06d}.npz')
        for pool,split,ids in [('val500','val','val_ids.txt'),('val100','val','monitor100_ids.txt'),('train16','train','atlas_train16_ids.txt')]:
            ev.metrics(RUN/f'decoded/epoch-120/{split}',CFG/ids,REPORT/pool)
        ev.atlas(RUN/'decoded/epoch-120/val',CFG/'atlas_val32_ids.txt',REPORT/'val-atlas')
        ev.atlas(RUN/'decoded/epoch-120/train',CFG/'atlas_train16_ids.txt',REPORT/'train-atlas')
        comparison={}
        for pool in ['val500','val100']:
            a=old.read(ROOT/f'reports/lr-half-from100/L/{pool}/metrics.json');b=old.read(REPORT/pool/'metrics.json')
            comparison[pool]=dict(L110=a,L120=b,delta={k:b['mean'][k]-v for k,v in a['mean'].items()},S_delta=b['S']-a['S'])
        old.save(REPORT/'comparison.json',comparison)
        for path,digest in old.read(REPORT/'protected-before.json').items():assert old.sha(ROOT/path)==digest,path
        (REPORT/'EVALUATION_COMPLETE').write_text('Scoring complete; AI atlas review and delivery pending.\n')
    (REPORT/(name+'.done')).write_text('exit0\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['generate','score']);parser.add_argument('--resume');args=parser.parse_args()
    if args.phase:phase(args.phase);return
    lock=(ROOT/'runtime/lr-half.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def event(**kw):
        with (ROOT/'logs/lr-half-to120-events.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
        print(json.dumps(kw),flush=True)
    common.event=event
    if not (RUN/'checkpoint-120/COMPLETE').exists():
        source=Path(args.resume).resolve() if args.resume else RUN/'checkpoint-110'
        state=old.read(source/'progress.json')
        assert source.parent==RUN and (source/'COMPLETE').exists() and 110<=state['completed_epochs']<120
        assert state['run_id']==old.L.name and state['current_lr']==[5e-5] and state['global_optimizer_updates']==state['completed_epochs']*532
        if not args.resume:assert not any(RUN.glob('checkpoint-11[1-9]*')), 'Inspect latest L COMPLETE; explicit --resume required'
        common.execute('lr-half-train-to-120',[sys.executable,'scripts/run_train.py','--branch-lr-half','--target','120','--resume',str(source)])
    for n in [115,120]:assert (RUN/f'checkpoint-{n}/COMPLETE').exists() and (RUN/f'exports/epoch-{n}/inference.pt').exists()
    for name in ['generate','score']:
        if not (REPORT/(name+'.done')).exists():common.execute('lr-half-to120-'+name,[sys.executable,__file__,'--phase',name])
    event(action='controller_complete',exit_code=0)
if __name__=='__main__':main()
