"""Bounded L120 correspondence control; no optimizer or training entry is used."""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import evaluate_stage as ev
import lr_half_evaluate as old
import stage3 as common
ROOT=ev.ROOT; RUN=old.L; REPORT=ROOT/'reports/brain-correspondence-l120'
WORK=RUN/'comparison/brain-correspondence-l120'
BASE=ROOT/'reports/lr-half-to120'
KEYS=['PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV']
MAP_SEED=2026092401;BOOT_SEED=2026092402

def prepare():
    assert not WORK.exists()
    full=old.ids('val_ids.txt');assert len(full)==len(set(full))==500
    ident=old.identity(RUN,120,63840,RUN/'decoded/epoch-120/val')
    assert ident['sha256']==old.read(BASE/'delivery-checks.json')['export_sha256']
    # Prior experiment verified this full preservation manifest at completion. Reuse its digests.
    assert (ROOT/'reports/l130-val500-followup/EVALUATION_COMPLETE').exists()
    protected=old.read(ROOT/'reports/l130-val500-followup/protected-before.json')
    for folder in [ROOT/'reports',ROOT/'configs']:
        for p in folder.rglob('*'):
            if p.is_file() and REPORT not in p.parents and str(p.relative_to(ROOT)) not in protected:
                protected[str(p.relative_to(ROOT))]=old.sha(p)
    for name in ['decode_brain_adapter.py','brain_adapter/dataset.py','metric_brain_adapter.py','scripts/evaluate_stage.py',
                 'train_brain_adapter.py','branch_lr_resume.py']:
        protected[name]=old.sha(ROOT/name)
    for p in [RUN/'epochs.jsonl',RUN/'checkpoint-140/progress.json']:
        assert old.sha(p)==protected[str(p.relative_to(ROOT))]
    old.save(REPORT/'protected-before.json',protected)
    rng=np.random.default_rng(MAP_SEED)
    while True:
        donors=rng.permutation(full)
        if all(i!=j for i,j in zip(full,donors)):break
    mapping=[dict(target_image_id=i,brain_source_image_id=int(j)) for i,j in zip(full,donors)]
    old.save(REPORT/'brain-map.json',mapping)
    WORK.mkdir(parents=True)
    from decode_brain_correspondence import remap_fields
    import torch
    target={'image_id':1,'img_encoder':torch.ones(3,2,2),'brain_lh_f':torch.ones(2,3),'brain_rh_f':torch.ones(2,3)*2}
    source={'image_id':2,'img_encoder':torch.zeros(3,2,2),'brain_lh_f':torch.ones(2,3)*3,'brain_rh_f':torch.ones(2,3)*4}
    result=remap_fields(target,source)
    assert result['image_id']==1 and result['img_encoder'] is target['img_encoder']
    assert result['brain_lh_f'] is source['brain_lh_f'] and result['brain_rh_f'] is source['brain_rh_f']
    assert target['brain_lh_f'].sum()==6
    old.save(REPORT/'preflight.json',dict(weight_identity=ident,mapping_seed=MAP_SEED,map_sha256=old.sha(REPORT/'brain-map.json'),
        mapped_images=500,fixed_points=0,source_coverage=500,remap_unit_checks_passed=True,
        correct_results_reused=str(BASE),training_updates_added=0,original_decode_sha256=protected['decode_brain_adapter.py'],
        original_metric_sha256=protected['metric_brain_adapter.py'],experiment_scripts_sha256={name:old.sha(ROOT/name) for name in ['scripts/brain_correspondence.py','scripts/decode_brain_correspondence.py']}))

def decode(mode,ids,out):
    c=old.read(ROOT/'configs/stage1.json')
    cmd=[sys.executable,str(ROOT/'scripts/decode_brain_correspondence.py'),
        '--control-mode',mode,'--control-map',str(REPORT/'brain-map.json'),'--no_selection','--split','val',
        '--image_ids',str(ROOT/'configs'/ids),'--decoded_stimuli_dir',str(out),'--model_weights_dir',str(RUN),
        '--saved_epochs','120','--num_predictions','2','--noise_factor','4','--precision','fp32','--batch_size','1',
        '--denoising_steps','50','--seed','42','--num_workers','0','--export_path',str(RUN/'exports/epoch-120/inference.pt')]
    for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk','condition_dim','num_decoder_queries','sub_approach']:
        cmd+=['--'+k,str(c[k])]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
    print(json.dumps(dict(decoder_command=cmd)),flush=True)
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)

def identity():
    decode('identity','decode2_ids.txt',WORK/'identity2')
    rows=[]
    for i in old.ids('decode2_ids.txt'):
        with np.load(WORK/f'identity2/image_{i:06d}.npz') as a,np.load(RUN/f'decoded/epoch-120/val/image_{i:06d}.npz') as b:
            for field in ['image_id','groundtruth_image','candidate_images','seeds']:assert np.array_equal(a[field],b[field]),(i,field)
        rows.append(dict(image_id=i,both_candidates_pixel_exact=True,gt_and_seeds_exact=True))
    old.save(REPORT/'identity-reproduction.json',dict(images=rows,passed=True))
    old.save(REPORT/'identity-input-audit.json',[json.loads(x) for x in (WORK/'identity2/brain-input-audit.jsonl').read_text().splitlines()])

def generate():
    assert old.read(REPORT/'identity-reproduction.json')['passed']
    decode('permuted','val_ids.txt',WORK/'permuted500')
    old.identity(RUN,120,63840,WORK/'permuted500')
    rows=[json.loads(x) for x in (WORK/'permuted500/brain-input-audit.jsonl').read_text().splitlines()]
    assert len(rows)==500 and len({x['target_image_id'] for x in rows})==500
    assert len({x['brain_source_image_id'] for x in rows})==500
    for row in rows:
        i=row['target_image_id'];assert i!=row['brain_source_image_id']
        assert row['actual_generator_input_sha256']==row['expected_input_sha256']!=row['target_input_sha256']
        old.verify_file(WORK/f'permuted500/image_{i:06d}.npz',i,RUN/f'decoded/epoch-120/val/image_{i:06d}.npz')
    old.save(REPORT/'brain-input-audit.json',rows)
    old.save(REPORT/'control-protocol.json',old.read(WORK/'permuted500/control-protocol.json'))

def score():
    for pool,ids in [('val500','val_ids.txt'),('val100','monitor100_ids.txt')]:
        ev.metrics(WORK/'permuted500',ROOT/'configs'/ids,REPORT/'permuted'/pool)
    ev.atlas(WORK/'permuted500',ROOT/'configs/atlas_val32_ids.txt',REPORT/'permuted/val-atlas')

def compare():
    full=old.ids('val_ids.txt');frames={};summaries={}
    for label,base in [('correct',BASE),('permuted',REPORT/'permuted')]:
        candidates=[pd.read_csv(base/f'val500/candidate-{k}.csv').set_index('image_id') for k in range(2)]
        for c in candidates:assert len(c)==500 and c.index.is_unique and set(c.index)==set(full)
        frames[label]=(candidates[0].loc[full,KEYS]+candidates[1].loc[full,KEYS])/2
        frames[label]['S']=frames[label][['Alex(5)','Incep','CLIP']].mean(axis=1)
        m=old.read(base/'val500/metrics.json')
        assert np.allclose(frames[label].mean(),[m['mean'][k] for k in KEYS]+[m['S']],atol=1e-10,rtol=0)
        summaries[label]={pool:old.read(base/pool/'metrics.json') for pool in ['val500','val100']}
    delta=frames['correct']-frames['permuted'];values=delta.to_numpy();assert np.isfinite(values).all()
    pd.concat([frames['correct'].add_prefix('correct_'),frames['permuted'].add_prefix('permuted_'),delta.add_prefix('delta_')],axis=1).to_csv(REPORT/'paired-correct-minus-permuted.csv')
    rng=np.random.default_rng(BOOT_SEED);samples=np.empty((10000,9))
    for start in range(0,10000,100):
        idx=rng.integers(0,500,size=(100,500));samples[start:start+100]=values[idx].mean(axis=1)
    ci=np.quantile(samples,[.025,.975],axis=0)
    rows=[dict(metric=k,delta=float(values[:,j].mean()),low=float(ci[0,j]),high=float(ci[1,j]),interval_level=.95,
        better='lower' if k in ('Eff','SwAV') else 'higher') for j,k in enumerate(KEYS+['S'])]
    for path,h in old.read(REPORT/'protected-before.json').items():assert old.sha(ROOT/path)==h,path
    old.save(REPORT/'comparison.json',dict(summaries=summaries,results=rows,resamples=10000,bootstrap_seed=BOOT_SEED,mapping_seed=MAP_SEED,
        interpretation='Exploratory image-paired descriptive bootstrap conditional on one fixed derangement and chosen L120; The same images serve as targets and donors under a bijection, coupling paired rows; no unconditional iid claim. Not a permutation-test p-value or independent-test/cross-seed inference.',
        images_changed=sum(not np.array_equal(np.load(WORK/f'permuted500/image_{i:06d}.npz')['candidate_images'],
                          np.load(RUN/f'decoded/epoch-120/val/image_{i:06d}.npz')['candidate_images']) for i in full)))
    old.save(REPORT/'completion.json',dict(training_updates_added=0,permuted_images=500,permuted_candidates=1000,
        identity_check_images=2,identity_check_candidates=4,original_outputs_unchanged=True,protected_hashes_unchanged=True))
    (REPORT/'EVALUATION_COMPLETE').write_text('Compute complete; fixed atlas AI review and Chinese report pending.\n')

def event(**kw):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    with (ROOT/'logs/brain-correspondence-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','identity','generate','score','compare']);args=parser.parse_args()
    if args.phase:
        globals()[args.phase]()
        (REPORT/f'{args.phase}.done').write_text('exit0\n')
        return
    lock=(ROOT/'runtime/brain-correspondence.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (REPORT/'PLAN.md').exists()
    common.event=event
    for phase in ['prepare','identity','generate','score','compare']:
        if (REPORT/f'{phase}.done').exists():continue
        common.execute('brain-correspondence-'+phase,[sys.executable,__file__,'--phase',phase])
    event(action='controller_complete',exit_code=0)

if __name__=='__main__':
    try:main()
    except Exception as e:event(action='failed',error=repr(e));raise
