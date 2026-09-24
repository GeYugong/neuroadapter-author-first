"""Bounded seed43 endpoint orchestration; reuse author and prior replication functions."""
import argparse,concurrent.futures,fcntl,json,os,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np
import hl_content_replication as rep
import hl_content_interchange as pilot
import lr_half_evaluate as old
from seed43_eval_config import ROOT,RUN,REPORT,WORK,STUDY
PRIOR=ROOT/'reports/hl-content-replication-l120'
GT=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100/decoded/epoch-120/val'
AA=WORK/'AA'

def event(**kw):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    with (ROOT/'logs/seed43-eval-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def execute(name,cmd,gpu=0):
    log=ROOT/f'logs/seed43-eval-{name}.log';assert not log.exists(), 'Inspect existing output/log; no blind replay'
    event(action='start',name=name,command=cmd,gpu=gpu)
    with log.open('w') as f:result=subprocess.run(cmd,cwd=ROOT,env=pilot.environment(gpu),stdout=f,stderr=subprocess.STDOUT)
    event(action='exit',name=name,exit_code=result.returncode)
    assert result.returncode==0,(name,result.returncode)

def check_ready():
    assert (ROOT/'reports/seed43-hl-replication/TRAINING_COMPLETE').exists()
    for e in (100,110,120):assert (RUN/f'checkpoint-{e}/COMPLETE').exists()
    state=old.read(RUN/'checkpoint-120/progress.json')
    assert (state['completed_epochs'],state['global_optimizer_updates'],state['skipped_updates'],state['current_lr'])==(120,63840,0,[5e-5])
    assert state['run_id']==RUN.name and state['config']['seed']==43
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'Wait for training and other GPU processes to exit'

def prepare():
    check_ready();assert not WORK.exists() and not (REPORT/'protocol.json').exists()
    m=old.read(RUN/'exports/epoch-120/metadata.json');digest=old.sha(RUN/'exports/epoch-120/inference.pt')
    assert (m['epoch'],m['global_optimizer_updates'],m['sha256'])==(120,63840,digest)
    assert m['config']['seed']==43
    locked=old.read(ROOT/'reports/evidence-freeze-seed43-plan/protocol-proposal.json')['target_donor_and_atlas_files_sha256']
    for rel,h in locked.items():assert old.sha(ROOT/rel)==h,rel
    REPORT.mkdir(parents=True,exist_ok=True)
    for name in ['target_ids.txt','atlas32_ids.txt']:shutil.copy2(PRIOR/name,REPORT/name)
    ids=np.loadtxt(REPORT/'target_ids.txt',dtype=int).tolist();assert len(ids)==100
    np.savetxt(REPORT/'check_ids.txt',ids[:1],fmt='%d')
    for batch in range(4):
        folder=REPORT if batch==0 else REPORT/f'donor-{batch}';folder.mkdir(exist_ok=True)
        source=PRIOR/f'donor-{max(batch,1)}/map.json'
        if batch:shutil.copy2(source,folder/'map.json')
        cfg=dict(study=STUDY,mode='formal',batch=batch,allowed_conditions=['AA'] if batch==0 else ['BA','AB','BB'],target_ids=str(REPORT/'target_ids.txt'),mapping_path=str(source),mapping_sha256=old.sha(source),model_sha256=digest)
        old.save(folder/'study-config.json',cfg)
    cfg=dict(study=STUDY,mode='check',batch=1,allowed_conditions=['AA','BB'],target_ids=str(REPORT/'check_ids.txt'),mapping_path=str(PRIOR/'donor-1/map.json'),mapping_sha256=old.sha(PRIOR/'donor-1/map.json'),model_sha256=digest,reference_root=str(WORK/'checks/reference'))
    old.save(REPORT/'check-config.json',cfg)
    protocol=old.read(PRIOR/'protocol.json')
    protocol.update(study=STUDY,formal_candidates=2000,check_candidate_cap=8,training_updates=63840,epoch=120,training_seed=43,decode_seed=42,check_target_ids=ids[:1],check_design='One fixed target x two candidates x AA/BB x original/content paths = eight generated candidates',limits='Second predesignated training seed at fixed120; same repeatedly used development targets/donors. Two seeds do not support seed-population inference; no independent test, brain causality or HL specificity. Mixed E may be off-distribution. Historical seed42 endpoint was selected adaptively, new endpoint fixed beforehand.')
    old.save(REPORT/'protocol.json',protocol)
    files=[RUN/'exports/epoch-120'/x for x in ['inference.pt','metadata.json','parcel_vertices.pt']]+[RUN/'epochs.jsonl',ROOT/'decode_brain_adapter.py',ROOT/'metric_brain_adapter.py',ROOT/'reports/ibbi-roi-l120-pilot/roi_mapping.json']
    files+=list(REPORT.glob('*.json'))+list(REPORT.glob('*.txt'))+list(REPORT.glob('donor-*/*.json'))
    # Only historical GT is used by atlas/direction, never its generated candidates as new AA.
    for i in protocol['permutation']:files.append(GT/f'image_{i:06d}.npz')
    old.save(REPORT/'input-lock.json',{str(p.relative_to(ROOT)):old.sha(p) for p in files})
    old.save(REPORT/'model-lock.json',dict(metadata=m,sha256=digest))
    WORK.mkdir()

def command(label,m=None,check=False,raw=False):
    batch=0 if m is None else m
    cmd=pilot.command(label,check)
    def setarg(k,v):cmd[cmd.index(k)+1]=str(v)
    setarg('--image_ids',REPORT/('check_ids.txt' if check else 'target_ids.txt'))
    folder=WORK/('checks/reference' if raw else 'checks/content') if check else (WORK if batch==0 else WORK/f'donor-{batch}')
    setarg('--decoded_stimuli_dir',folder/label);setarg('--model_weights_dir',RUN);setarg('--export_path',RUN/'exports/epoch-120/inference.pt')
    cfg=REPORT/'check-config.json' if check else (REPORT if batch==0 else REPORT/f'donor-{batch}')/'study-config.json'
    if raw:
        cmd[1]=str(ROOT/'scripts/decode_brain_correspondence.py')
        at=cmd.index('--content-condition');cmd[at:at+2]=['--control-mode','identity' if label=='AA' else 'permuted']
        cmd+=['--control-map',str(PRIOR/'donor-1/map.json')]
    cmd+=['--study-config',str(cfg)]
    return cmd

def identity():
    records=[];i=int(np.loadtxt(REPORT/'check_ids.txt',dtype=int))
    for label in ['AA','BB']:
        execute('reference-'+label,command(label,check=True,raw=True))
        execute('identity-'+label,command(label,check=True))
        a=WORK/f'checks/reference/{label}/image_{i:06d}.npz';b=WORK/f'checks/content/{label}/image_{i:06d}.npz'
        for p in [a,b]:old.verify_file(p,i,GT/p.name)
        with np.load(a) as za,np.load(b) as zb:
            for k in ['image_id','candidate_images','groundtruth_image','seeds']:assert np.array_equal(za[k],zb[k]),(label,k)
        records.append(dict(condition=label,image_id=i,both_candidates_pixel_exact=True,gt_and_seed_exact=True))
    old.save(REPORT/'noninterference.json',dict(passed=True,new_candidates=8,checks=records))

def baseline():
    assert old.read(REPORT/'noninterference.json')['passed']
    execute('AA',command('AA'))
    old.identity(RUN,120,63840,AA)
    assert len(list(AA.glob('image_*.npz')))==100
    for i in rep.targets():old.verify_file(AA/f'image_{i:06d}.npz',i,GT/f'image_{i:06d}.npz')

def direction():
    for m in range(1,4):pilot.direction_for(rep.targets(),rep.mapping(m),rep.sources(m),REPORT/f'donor-{m}',GT)

def atlas():
    ids=np.loadtxt(REPORT/'atlas32_ids.txt',dtype=int).tolist();assert ids==rep.targets()[:32]
    lines=['# Seed43固定32目标：三批供体、全部候选','']
    for m in range(1,4):
        pilot.atlas_for(ids,rep.mapping(m),rep.sources(m),REPORT/f'donor-{m}/atlas',GT)
        for k in range(2):
            for page in range(1,5):lines.append(f'- [供体{m} 候选{k} 第{page}页](donor-{m}/atlas/candidate-{k}-page-{page}.jpg)')
    (REPORT/'ATLAS.md').write_text('\n'.join(lines)+'\n')

def finish():
    checks={p:old.sha(ROOT/p)==h for p,h in old.read(REPORT/'input-lock.json').items()}
    for p,h in old.read(ROOT/'reports/seed43-hl-replication/protected-before.json').items():checks[p]=old.sha(ROOT/p)==h
    for p,h in old.read(PRIOR/'input-lock.json').items():checks[p]=old.sha(ROOT/p)==h
    assert all(checks.values())
    count=len(list(AA.glob('image_*.npz')))*2+sum(len(list((WORK/f'donor-{m}'/c).glob('image_*.npz')))*2 for m in range(1,4) for c in ['BA','AB','BB'])
    check_count=len(list((WORK/'checks').glob('*/*/image_*.npz')))*2
    assert count==2000 and check_count==8
    rows=[json.loads(x) for x in (RUN/'epochs.jsonl').read_text().splitlines()]
    assert len(rows)==120 and [x['epoch'] for x in rows]==list(range(1,121))
    assert all(x['global_optimizer_updates']==x['epoch']*532 for x in rows)
    old.save(REPORT/'completion.json',dict(formal_candidates=count,check_candidates=check_count,training_epochs=120,actual_optimizer_updates=63840,input_hashes_unchanged=checks,compute_complete=True,AI_review_and_publication_pending=True))
    (REPORT/'EVALUATION_COMPLETE').write_text('Calculation complete; actual 24-page AI review and publication still pending.\n')

def configure():
    rep.RUN=RUN;rep.REPORT=REPORT;rep.WORK=WORK;rep.AA=AA;rep.execute=execute;rep.command=command

def main():
    configure()
    phases={'prepare':prepare,'identity':identity,'baseline':baseline,'generate':rep.generate,'score':rep.score,'direction':direction,'compare':rep.compare,'atlas':atlas,'finish':finish}
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=list(phases));args=parser.parse_args()
    if args.phase:
        phases[args.phase]();(REPORT/(args.phase+'.done')).write_text('exit0\n');return
    check_ready()
    lock=(ROOT/'runtime/seed43-eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in phases:
        if (REPORT/(name+'.done')).exists():continue
        execute(name,[sys.executable,__file__,'--phase',name])
    event(action='controller_complete',exit_code=0)
if __name__=='__main__':
    try:main()
    except Exception as exc:event(action='failed',error=repr(exc));raise
