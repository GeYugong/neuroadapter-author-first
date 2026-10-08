"""Exact L450 restore; optimizer and RNG unchanged; new objective only."""
import copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from branch_lr_resume import equal
ROOT=Path(__file__).resolve().parent
SOURCE='subject01-author-fp32-seed42-lr1e-5-from400'
BRANCH='subject01-author-fp32-seed42-lr1e-5-spatiall1-from450'

def validate(args):
    source=Path(args.resume_from).resolve();out=Path(args.output_dir).resolve()
    assert args.run_id==BRANCH and out==ROOT/'runs'/BRANCH
    assert args.num_train_epochs in (475,500) and args.learning_rate==1e-5 and args.weight_decay==1e-6
    assert source==ROOT/'runs'/SOURCE/'checkpoint-450' or source.parent==out
    baseline=json.loads((ROOT/'configs/stage1.json').read_text())
    for key,value in baseline.items():
        if key not in ('run_id','output_dir','learning_rate','weight_decay'):assert getattr(args,key)==value,key
    state=json.loads((source/'progress.json').read_text())
    assert (source/'COMPLETE').exists() and 450<=state['completed_epochs']<args.num_train_epochs
    assert source.name==f"checkpoint-{state['completed_epochs']}"
    assert state['skipped_updates']==0 and state['global_optimizer_updates']==state['completed_epochs']*532
    assert state['run_id']==(SOURCE if source.parent.name==SOURCE else BRANCH)
    assert state['current_lr']==[1e-5]
    own=[int(p.name.split('-')[-1]) for p in out.glob('checkpoint-*') if (p/'COMPLETE').exists()]
    if own:assert source.parent==out and state['completed_epochs']==max(own),'Use latest COMPLETE own checkpoint'


def restore(args,accelerator,optimizer,models,loader,dataset,progress):
    validate(args);source=Path(args.resume_from);initial=source.parent.name==SOURCE
    assert progress.run_id==(SOURCE if initial else BRANCH)
    auditpath=Path(args.output_dir,f'branch-resume-{progress.completed_epochs}-rank{accelerator.process_index}.json')
    assert not auditpath.exists(),'Do not overwrite a prior startup audit; inspect attempt first'
    saved_rng=torch.load(source/f'random_states_{accelerator.process_index}.pkl',map_location='cpu',weights_only=False)
    equal(random.getstate(),saved_rng['random_state']);equal(np.random.get_state(),saved_rng['numpy_random_seed'])
    equal(torch.get_rng_state(),saved_rng['torch_manual_seed']);equal(torch.cuda.get_rng_state_all(),saved_rng['torch_cuda_manual_seed'])
    for i,model in enumerate(models):
        saved=torch.load(source/('pytorch_model.bin' if i==0 else f'pytorch_model_{i}.bin'),map_location='cpu',weights_only=False)
        equal(saved,accelerator.unwrap_model(model).state_dict());del saved
    saved=torch.load(source/'optimizer.bin',map_location='cpu',weights_only=False);equal(saved,optimizer.state_dict())
    assert all(g['lr']==1e-5 and g['weight_decay']==1e-6 for g in optimizer.param_groups)
    equal(saved,optimizer.state_dict());equal(loader.generator.get_state(),progress.loader_generator_state)
    loader.set_epoch(progress.completed_epochs)
    sampler=copy.deepcopy(loader.batch_sampler,{id(dataset):dataset});batches=list(sampler)
    ids=np.asarray(dataset.base_dataset.split_presented_imgs)[np.asarray(batches)]
    gathered=accelerator.gather(torch.tensor(ids,device=accelerator.device)).cpu().numpy()
    ordered=gathered.reshape(accelerator.num_processes,len(batches),-1).transpose(1,0,2).reshape(-1)
    digest=hashlib.sha256(ordered.astype(np.int64).tobytes()).hexdigest()
    reference={x['epoch']:x for x in map(json.loads,(ROOT/'runs'/SOURCE/'epochs.jsonl').read_text().splitlines())}
    assert digest==reference[progress.completed_epochs+1]['image_order_sha256']
    equal(random.getstate(),saved_rng['random_state']);equal(np.random.get_state(),saved_rng['numpy_random_seed'])
    equal(torch.get_rng_state(),saved_rng['torch_manual_seed']);equal(torch.cuda.get_rng_state_all(),saved_rng['torch_cuda_manual_seed'])
    record=dict(source_run=SOURCE,source_checkpoint=str(ROOT/'runs'/SOURCE/'checkpoint-450'),resume_source=str(source),branch_run_id=BRANCH,rank=accelerator.process_index,completed_epochs=progress.completed_epochs,global_optimizer_updates=progress.global_optimizer_updates,model_exact=True,adamw_before_exact=True,adamw_unchanged=True,rng_exact=True,loader_restored=True,optimizer_lr=[g['lr'] for g in optimizer.param_groups],optimizer_weight_decay=[g['weight_decay'] for g in optimizer.param_groups],next_epoch_order_sha256=digest,order_matches_historical=True,continuation_target=args.num_train_epochs)
    auditpath.write_text(json.dumps(record,indent=2)+'\n')
    progress.run_id=BRANCH;progress.current_lr=[1e-5];progress.branch_provenance=record
    progress.code_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    accelerator.wait_for_everyone();print('BALANCED_RESUME_AUDIT '+json.dumps(record),flush=True)
