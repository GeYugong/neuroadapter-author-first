"""Sequential process orchestration through the authorized cumulative epoch 60; one milestone per invocation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/stage1.json').read_text())
RUN=Path(CONFIG['output_dir'])

def event(**fields):
    record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**fields)
    with (ROOT/'logs/stage3-events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)

def execute(label,cmd):
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPU compute processes already active; not starting: '+active)
    event(action='launch',label=label,command=cmd,parent_pid=os.getpid())
    with (ROOT/'logs'/f'{label}.log').open('a') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
        event(action='running',label=label,pid=proc.pid)
        code=proc.wait()
    event(action='exit',label=label,pid=proc.pid,exit_code=code)
    if code:raise RuntimeError(f'{label} failed with exit {code}; see logs/{label}.log')

def summarize(epoch):
    report=ROOT/'reports'/f'epoch-{epoch}'
    if epoch==1:
        reload_check=RUN/'decoded/epoch-1/reload_check.json'
        assert reload_check.exists()
        report.mkdir(parents=True,exist_ok=True)
        shutil.copy2(reload_check,report/'reload_check.json')
        (report/'COMPLETE').write_text('Epoch 1 decode and export reload checked.\n')
        return
    val=json.loads((report/'val100/metrics.json').read_text())
    train=json.loads((report/'train16/metrics.json').read_text())
    text=f'# Epoch {epoch}\n\n同一固定100张历史开发验证图，每图2候选分别评分后平均；没有候选选择。训练16图仅为辅助观察，二选一负例池不同，不与验证识别率直接相减。\n\n'
    keys=list(val['mean'])
    text+='| 集合 | '+ ' | '.join(keys)+' | S |\n|---|'+ '|'.join(['---']*(len(keys)+1))+'|\n'
    for name,result in [('val100',val),('train16',train)]:
        text+='| '+name+' | '+' | '.join(f'{result["mean"][k]:.5f}' for k in keys)+f' | {result["S"]:.5f} |\n'
    previous=[]
    for e in [5,10,20,30,40,50,60]:
        f=ROOT/f'reports/epoch-{e}/val100/metrics.json'
        if e <= epoch and (f.parent.parent/'COMPLETE').exists():previous.append((e,json.loads(f.read_text())))
    if len(previous)>1:
        before=previous[-2][1]
        text+='\n相同100图相对前一阶段变化：'+ '；'.join(f'{k} {val["mean"][k]-before["mean"][k]:+.5f}' for k in keys)+'。Eff/SwAV越低越好，其余越高越好。\n'
    text+='\n图册按预先固定ID展示GT和两个候选，不按效果筛选。\n'
    for split,n in [('val',4),('train',2)]:
        for i in range(1,n+1):text+=f'\n![{split} atlas {i}]({split}-atlas/atlas-{i}.jpg)\n'
    best_epoch,best=max(previous,key=lambda p:p[1]['S'])
    best_dir=RUN/'best_so_far';best_dir.mkdir(exist_ok=True)
    source=RUN/'exports'/f'epoch-{best_epoch}'
    for name in ['inference.pt','metadata.json','parcel_vertices.pt']:shutil.copy2(source/name,best_dir/name)
    backup=ROOT/'backup'/CONFIG['run_id']/'best_so_far';backup.mkdir(parents=True,exist_ok=True)
    for name in ['inference.pt','metadata.json','parcel_vertices.pt']:shutil.copy2(source/name,backup/name)
    assert hashlib.sha256((best_dir/'inference.pt').read_bytes()).digest()==hashlib.sha256((backup/'inference.pt').read_bytes()).digest()
    selection={'epoch':best_epoch,'criterion':'maximum fixed val100 S among completed stages; provisional development selection',
               'S':best['S'],'full_checkpoint':str(RUN/f'checkpoint-{best_epoch}'),
               'sha256':hashlib.sha256((best_dir/'inference.pt').read_bytes()).hexdigest(),
               'backup':str(backup),'backup_note':'Two server-local copies on same filesystem, not an offsite backup'}
    (ROOT/'reports/best_so_far.json').write_text(json.dumps(selection,indent=2))
    text+=f'\n当前按固定val100 S保留的best_so_far为epoch {best_epoch}，仅为本阶段开发选择；所有里程碑完整状态均保留。科学设置未调整。\n'
    if epoch==60:text+='\n已到本次授权上限60，训练和评价完成后停止，等待用户决定是否继续。\n'
    (report/'README.md').write_text(text)
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write(f'\nEpoch {epoch}阶段完成：val100 S={val["S"]:.5f}；best_so_far epoch={best_epoch}。实际八项指标与固定图册见reports/epoch-{epoch}。\n')

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',type=int,choices=[60],required=True)
    target=parser.parse_args().target
    lock=(ROOT/'runtime/stage3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (ROOT/'reports/STAGE2_COMPLETE').exists()
    previous=target-10
    checkpoint=RUN/f'checkpoint-{previous}'
    assert (checkpoint/'COMPLETE').exists()
    assert (ROOT/f'reports/epoch-{previous}/COMPLETE').exists()
    state=json.loads((checkpoint/'progress.json').read_text())
    assert state['completed_epochs']==previous and state['global_optimizer_updates']==previous*532
    if any(int(p.name.split('-')[1])>previous for p in RUN.glob('checkpoint-*') if p.name.split('-')[1].isdigit()):
        raise RuntimeError('Later checkpoint exists; inspect before recovery, do not replay completed epochs')
    if (ROOT/f'reports/epoch-{target}').exists():raise FileExistsError('Target report exists')
    event(action='start',pid=os.getpid(),target=target,max_cumulative_epoch=60,source=str(checkpoint))
    execute(f'train-to-{target}',[sys.executable,'scripts/run_train.py','--stage3','--target',str(target),'--resume',str(checkpoint)])
    state=json.loads((RUN/f'checkpoint-{target}/progress.json').read_text())
    assert state['completed_epochs']==target
    execute(f'eval-{target}',[sys.executable,'scripts/evaluate_stage.py','--run',str(RUN),'--epoch',str(target)])
    summarize(target)
    event(action='stage_complete',epoch=target,updates=state['global_optimizer_updates'])
    if target==60:
        event(action='completed_stopped',epoch=60)
        (ROOT/'reports/STAGE3_COMPLETE').write_text('Cumulative epoch 60 completed; no further training authorized.\n')

if __name__=='__main__':
    try:main()
    except Exception as e:
        event(action='failed',error=repr(e))
        raise
