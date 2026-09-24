"""Explicit seed43 initialization and own-run full-state continuation audit."""
import copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from branch_lr_resume import equal
ROOT=Path(__file__).resolve().parent
RUN_ID='subject01-author-fp32-seed43-lrhalf-at100'
RUN=ROOT/'runs'/RUN_ID

def validate(args):
    assert args.seed43_replication and args.run_id==RUN_ID and Path(args.output_dir).resolve()==RUN
    assert args.num_train_epochs in (100,120) and args.seed==43
    assert args.learning_rate==(1e-4 if args.num_train_epochs==100 else 5e-5)
    baseline=json.loads((ROOT/'configs/stage1.json').read_text())
    for key,value in baseline.items():
        if key not in ('run_id','output_dir','seed','learning_rate'):
            assert getattr(args,key)==value, f'Seed43 science changed: {key}'
    assert args.train_ids is None and args.audit
    if not args.resume_from:
        assert args.num_train_epochs==100
        assert not any(RUN.glob('checkpoint-*')), 'Existing run requires explicit complete resume'
        return
    source=Path(args.resume_from).resolve()
    assert source.parent==RUN and (source/'COMPLETE').exists()
    state=json.loads((source/'progress.json').read_text())
    epoch=state['completed_epochs']
    assert state['run_id']==RUN_ID and source.name==f'checkpoint-{epoch}'
    assert (1<=epoch<100 if args.num_train_epochs==100 else 100<=epoch<120)
    assert state['global_optimizer_updates']==epoch*532 and state['skipped_updates']==0
    assert state['current_lr']==[1e-4 if epoch<=100 else 5e-5]
    for key,value in baseline.items():
        if key not in ('run_id','output_dir','seed','learning_rate'):
            assert state['config'][key]==value
    assert state['config']['seed']==43

def snapshot_rng():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())

def write_audit(args,accelerator,name,record):
    record.update(rank=accelerator.process_index,run_id=RUN_ID,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (RUN/f'{name}-rank{accelerator.process_index}.json').write_text(json.dumps(record,indent=2)+'\n')
    accelerator.wait_for_everyone()
    print('SEED43_AUDIT '+json.dumps(record),flush=True)

def audit_initial(args,accelerator,optimizer,models,loader,dataset,progress):
    validate(args);rng=snapshot_rng()
    assert accelerator.num_processes==2 and accelerator.mixed_precision=='no'
    assert progress.completed_epochs==0 and progress.global_optimizer_updates==0
    assert len(optimizer.state_dict()['state'])==0 and all(g['lr']==1e-4 for g in optimizer.param_groups)
    h=hashlib.sha256()
    for model in models:
        for name,p in accelerator.unwrap_model(model).named_parameters():
            if p.requires_grad:
                assert p.dtype==torch.float32
                h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    digests=accelerator.gather(torch.tensor(list(h.digest()),dtype=torch.uint8,device=accelerator.device)).cpu().reshape(2,-1)
    assert torch.equal(digests[0],digests[1]), 'Initial trainable parameters differ across ranks'
    reference=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100/exports/epoch-120/parcel_vertices.pt'
    equal(dataset.get_selected_voxel_indices(),torch.load(reference,map_location='cpu',weights_only=False))
    old=torch.load(ROOT/'runs/subject01-author-fp32-seed42/exports/epoch-0/inference.pt',map_location='cpu',weights_only=False)
    guidance=accelerator.unwrap_model(models[1]).state_dict()
    different=any(not torch.equal(v.cpu(),old['guidance_generator'][k].cpu()) for k,v in guidance.items())
    assert different, 'New seed must initialize its own guidance model'
    del old
    equal(rng,snapshot_rng())
    write_audit(args,accelerator,'initial',dict(seed=43,completed_epochs=0,global_optimizer_updates=0,optimizer_lr=[g['lr'] for g in optimizer.param_groups],optimizer_state_empty=True,trainable_parameters_fp32=True,trainable_sha256=h.hexdigest(),ranks_identical=True,guidance_differs_from_seed42_initialization=different,parcel_vertices_exactly_match_L120=True,rng_unchanged_by_audit=True,loader_generator_sha256=hashlib.sha256(loader.generator.get_state().numpy().tobytes()).hexdigest()))

def restore(args,accelerator,optimizer,models,loader,progress):
    validate(args);source=Path(args.resume_from);epoch=progress.completed_epochs
    assert progress.run_id==RUN_ID and progress.global_optimizer_updates==epoch*532
    saved_rng=torch.load(source/f'random_states_{accelerator.process_index}.pkl',map_location='cpu',weights_only=False)
    equal(snapshot_rng(),(saved_rng['random_state'],saved_rng['numpy_random_seed'],saved_rng['torch_manual_seed'],saved_rng['torch_cuda_manual_seed']))
    for i,model in enumerate(models):
        saved=torch.load(source/('pytorch_model.bin' if i==0 else f'pytorch_model_{i}.bin'),map_location='cpu',weights_only=False)
        equal(saved,accelerator.unwrap_model(model).state_dict());del saved
    equal(loader.generator.get_state(),progress.loader_generator_state)
    saved=torch.load(source/'optimizer.bin',map_location='cpu',weights_only=False)
    equal(saved,optimizer.state_dict())
    before_lr=[g['lr'] for g in optimizer.param_groups]
    assert before_lr==[1e-4 if epoch<=100 else 5e-5]
    expected=copy.deepcopy(saved)
    for group in optimizer.param_groups:group['lr']=args.learning_rate
    for group in expected['param_groups']:group['lr']=args.learning_rate
    equal(expected,optimizer.state_dict())
    equal(snapshot_rng(),(saved_rng['random_state'],saved_rng['numpy_random_seed'],saved_rng['torch_manual_seed'],saved_rng['torch_cuda_manual_seed']))
    progress.current_lr=[g['lr'] for g in optimizer.param_groups]
    write_audit(args,accelerator,f'resume-{epoch}-to-{args.num_train_epochs}',dict(source=str(source),completed_epochs=epoch,global_optimizer_updates=progress.global_optimizer_updates,model_exact=True,adamw_before_exact=True,adamw_after_only_authorized_lr_change=True,adamw_state_entries=len(saved['state']),lr_before=before_lr,lr_after=progress.current_lr,lr_change_location='after full Accelerate load and exact state audit; before next training epoch',dual_rank_rng_exact=True,loader_generator_exact=True,seed_not_reset_after_load=True))
