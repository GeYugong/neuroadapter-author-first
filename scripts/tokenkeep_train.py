"""Scoped entry reusing original science; restore, telemetry and exports are isolated."""
import sys,os,json,time,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import train_brain_adapter as t
import branch_lr_resume
import tokenkeep_resume as balanced_resume
import tokenkeep_objective as tokenkeep

def install(args):
    branch_lr_resume.validate=balanced_resume.validate;branch_lr_resume.restore=balanced_resume.restore
    state={'loss':[],'kept':[],'timestep_counts':[0]*4,'groups':{},'before':None}
    original_setup=t.setup_accelerator
    def setup(a):
        acc=original_setup(a);state['acc']=acc
        original_clip=acc.clip_grad_norm_
        def clip(params,max_norm,*pa,**kw):
            norms={}
            for name,group in state['groups'].items():
                terms=[p.grad.detach().float().norm().square() for p in group if p.grad is not None]
                norms[name]=torch.stack(terms).sum().sqrt().item() if terms else 0.
            state['group_grad_norm']=norms
            result=original_clip(params,max_norm,*pa,**kw);state['grad_norm']=float(result)
            return result
        acc.clip_grad_norm_=clip;return acc
    t.setup_accelerator=setup
    original_models=t.setup_models_and_optimizer
    def models(*pa,**kw):
        result=original_models(*pa,**kw);model,guidance,opt,_,_=result;acc=state['acc']
        raw=acc.unwrap_model(model)
        state['groups']={'mapper':list(guidance.parameters()),'projection':list(raw.image_proj_model.parameters()),'ip_kv':list(raw.adapter_modules.parameters())}
        original_step=opt.step
        def step(*a,**k):
            if not acc.sync_gradients:return original_step(*a,**k)
            update=args.progress.global_optimizer_updates+1
            snapshot=update==int(Path(args.resume_from).name.split('-')[-1])*532+1 or update%100==0
            before={name:[p.detach().clone() for p in group] for name,group in state['groups'].items()} if snapshot else None
            ret=original_step(*a,**k)
            ratios={}
            if snapshot:
                for name,group in state['groups'].items():
                    delta=sum(float((p.detach()-old).float().norm().square()) for p,old in zip(group,before[name]))
                    scale=sum(float(old.float().norm().square()) for old in before[name])
                    ratios[name]=(delta/max(scale,1e-30))**.5
                if update==int(Path(args.resume_from).name.split('-')[-1])*532+1:
                    assert all(v>0 for v in ratios.values()),ratios
                    assert all(v>0 for v in state['group_grad_norm'].values()),state['group_grad_norm']
            record=dict(update=update,rank=acc.process_index,mean_loss=sum(state['loss'])/len(state['loss']),microbatches=len(state['loss']),lr=[g['lr'] for g in opt.param_groups],skipped=bool(acc.optimizer_step_was_skipped),preclip_grad_norm=state['grad_norm'],clip_active=state['grad_norm']>args.clip_max_norm,group_preclip_grad_norm=state['group_grad_norm'],update_to_weight_norm=ratios,mean_kept_tokens=sum(state['kept'])/len(state['kept']),timestep_counts=state['timestep_counts'])
            with (Path(args.output_dir)/f'updates-rank{acc.process_index}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if snapshot:print('BALANCED_UPDATE '+json.dumps(record),flush=True)
            state['loss']=[];state['kept']=[];state['timestep_counts']=[0]*4
            return ret
        opt.step=step;return result
    t.setup_models_and_optimizer=models
    original_dropout=t.apply_fmri_token_dropout
    def dropout(tokens):
        rng=torch.get_rng_state();result=original_dropout(tokens)
        gen=torch.Generator().set_state(rng);b,n,_=tokens.shape
        mask=tokenkeep.mask(b,n,generator=gen)
        state['kept']+=mask.sum(1).tolist();return result
    t.apply_fmri_token_dropout=dropout
    original_weight=t.min_snr_loss_weights
    def weight(ts,*pa,**kw):
        for value in ts.detach().cpu().tolist():state['timestep_counts'][min(value//250,3)]+=1
        return original_weight(ts,*pa,**kw)
    t.min_snr_loss_weights=weight
    original_batch=t.process_training_batch
    def batch(*pa,**kw):
        loss=original_batch(*pa,**kw);state['loss'].append(float(loss.detach()));return loss
    t.process_training_batch=batch
    original_export=t.export_inference
    def export(*pa,**kw):
        if pa[-1] in (500,):return original_export(*pa,**kw)
    t.export_inference=export
    def cleanup(directory,keep_last_n=2):
        directory=Path(directory).resolve()
        assert directory==ROOT/'runs'/balanced_resume.BRANCH
        actual=json.loads((directory/'epochs.jsonl').read_text().splitlines()[-1])
        reference={v['epoch']:v for v in map(json.loads,(ROOT/'runs'/balanced_resume.SOURCE/'epochs.jsonl').read_text().splitlines())}
        assert actual['image_order_sha256']==reference[actual['epoch']]['image_order_sha256'], 'Actual epoch order differs from same-source control'
        cps=sorted([p for p in directory.glob('checkpoint-*') if (p/'COMPLETE').exists()],key=lambda p:int(p.name.split('-')[-1]))
        for cp in cps[:-keep_last_n]:
            if int(cp.name.split('-')[-1]) not in (500,):
                assert cp.resolve().parent==directory
                shutil.rmtree(cp)
        if shutil.disk_usage(ROOT).free<15*2**30:raise RuntimeError('Complete checkpoint saved; disk reserve low')
    t.cleanup_checkpoints=cleanup

def main():
    args=t.create_argument_parser().parse_args()
    assert args.branch_lr_half and not args.seed43_replication
    args.local_rank=int(os.environ.get('LOCAL_RANK',-1));args.time=args.run_id
    balanced_resume.validate(args)
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    args.token_keep_min=tokenkeep.KEEP_MIN
    t.apply_fmri_token_dropout=tokenkeep.apply
    install(args);t.main(args)

if __name__=='__main__':main()
