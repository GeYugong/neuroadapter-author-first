"""Bounded L continuation controller (120/130/140); existing author training and evaluation functions."""
import argparse,fcntl,json,sys
from pathlib import Path
import stage3 as common
import lr_half_evaluate as old
import evaluate_stage as ev
ROOT=common.ROOT;RUN=old.L;REPORT=ROOT/'reports/lr-half-to120';CFG=ROOT/'configs'
TARGET=120
def phase(name):
    epoch=TARGET
    val_ids="monitor100_ids.txt" if epoch==130 else "val_ids.txt"
    if name=='generate':
        state=old.read(RUN/f'checkpoint-{epoch}/progress.json')
        assert (state['completed_epochs'],state['global_optimizer_updates'],state['current_lr'])==(epoch,epoch*532,[5e-5])
        assert not (RUN/f'decoded/epoch-{epoch}').exists()
        ev.decode(RUN,epoch,'val',CFG/val_ids,RUN/f'decoded/epoch-{epoch}/val')
        ev.decode(RUN,epoch,'train',CFG/'atlas_train16_ids.txt',RUN/f'decoded/epoch-{epoch}/train')
    elif name=='score':
        old.identity(RUN,epoch,epoch*532,RUN/f'decoded/epoch-{epoch}/val')
        for i in old.ids(val_ids):old.verify_file(RUN/f'decoded/epoch-{epoch}/val/image_{i:06d}.npz',i,RUN/f'decoded/epoch-110/val/image_{i:06d}.npz')
        for pool,split,ids in [('val500','val','val_ids.txt'),('val100','val','monitor100_ids.txt'),('train16','train','atlas_train16_ids.txt')]:
            if epoch==130 and pool=='val500':continue
            ev.metrics(RUN/f'decoded/epoch-{epoch}/{split}',CFG/ids,REPORT/pool)
        ev.atlas(RUN/f'decoded/epoch-{epoch}/val',CFG/'atlas_val32_ids.txt',REPORT/'val-atlas')
        ev.atlas(RUN/f'decoded/epoch-{epoch}/train',CFG/'atlas_train16_ids.txt',REPORT/'train-atlas')
        comparison={}
        references={'L110':ROOT/'reports/lr-half-from100/L','L120':ROOT/'reports/lr-half-to120'} if epoch>120 else {'L110':ROOT/'reports/lr-half-from100/L'}
        if epoch==140:references['L130']=ROOT/'reports/lr-half-to140/epoch-130'
        for pool in (['val100'] if epoch==130 else ['val500','val100']):
            b=old.read(REPORT/pool/'metrics.json');comparison[pool]={'current':b,'references':{}}
            for label,base in references.items():
                if not (base/pool/'metrics.json').exists():continue
                a=old.read(base/pool/'metrics.json')
                comparison[pool]['references'][label]=dict(metrics=a,delta={k:b['mean'][k]-v for k,v in a['mean'].items()},S_delta=b['S']-a['S'])
        if epoch==120:
            comparison={pool:dict(L110=v['references']['L110']['metrics'],L120=v['current'],delta=v['references']['L110']['delta'],S_delta=v['references']['L110']['S_delta']) for pool,v in comparison.items()}
        old.save(REPORT/'comparison.json',comparison)
        for path,digest in old.read(REPORT/'protected-before.json').items():assert old.sha(ROOT/path)==digest,path
        (REPORT/'EVALUATION_COMPLETE').write_text('Scoring complete; AI atlas review and delivery pending.\n')
    (REPORT/(name+'.done')).write_text('exit0\n')
def main():
    global TARGET,REPORT
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['generate','score']);parser.add_argument('--resume');parser.add_argument('--target',type=int,choices=[120,130,140],default=120);args=parser.parse_args()
    TARGET=args.target;epoch=TARGET
    if epoch>120:REPORT=ROOT/f'reports/lr-half-to140/epoch-{epoch}'
    REPORT.mkdir(parents=True,exist_ok=True)
    if epoch==140:assert (ROOT/'reports/lr-half-to140/epoch-130/PUBLISHED_CONTINUE').exists(), 'Publish and review L130 before L140' 
    if args.phase:phase(args.phase);return
    lock=(ROOT/'runtime/lr-half.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def event(**kw):
        with (ROOT/f'logs/lr-half-to{epoch}-events.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
        print(json.dumps(kw),flush=True)
    common.event=event
    if not (RUN/f'checkpoint-{epoch}/COMPLETE').exists():
        source=Path(args.resume).resolve() if args.resume else RUN/f'checkpoint-{epoch-10}'
        state=old.read(source/'progress.json')
        assert source.parent==RUN and (source/'COMPLETE').exists() and epoch-10<=state['completed_epochs']<epoch
        assert state['run_id']==old.L.name and state['current_lr']==[5e-5] and state['global_optimizer_updates']==state['completed_epochs']*532
        if not args.resume:assert not any(int(p.name.split('-')[1].split('.')[0])>epoch-10 for p in RUN.glob('checkpoint-*')), 'Inspect latest L COMPLETE; explicit --resume required'
        common.execute(f'lr-half-train-to-{epoch}',[sys.executable,'scripts/run_train.py','--branch-lr-half','--target',str(epoch),'--resume',str(source)])
    for n in ([115,120] if epoch==120 else [epoch-10,epoch]):assert (RUN/f'checkpoint-{n}/COMPLETE').exists() and (RUN/f'exports/epoch-{n}/inference.pt').exists()
    for name in ['generate','score']:
        if not (REPORT/(name+'.done')).exists():common.execute(f'lr-half-to{epoch}-'+name,[sys.executable,__file__,'--target',str(epoch),'--phase',name])
    event(action='controller_complete',exit_code=0)
if __name__=='__main__':main()
