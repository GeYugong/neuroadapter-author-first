"""Narrow full-state audit/LR override for the single authorized branch."""
import copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
MAIN='subject01-author-fp32-seed42'
BRANCH=MAIN+'-lr5e-5-from100'
def equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    else: assert a==b

def validate(args):
    source=Path(args.resume_from).resolve();out=Path(args.output_dir).resolve()
    assert args.run_id==BRANCH and out==ROOT/'runs'/BRANCH
    assert args.num_train_epochs in (110,120,130,140,300) and args.learning_rate==5e-5
    if args.num_train_epochs>=120: assert source.parent==out, "L continuation requires own checkpoint"
    assert source==ROOT/'runs'/MAIN/'checkpoint-100' or source.parent==out
    baseline=json.loads((ROOT/'configs/stage1.json').read_text())
    for key,value in baseline.items():
        if key not in ('run_id','output_dir','learning_rate'): assert getattr(args,key)==value, f'Branch science changed: {key}'
    state=json.loads((source/'progress.json').read_text())
    lower = 140 if args.num_train_epochs == 300 else args.num_train_epochs - 10
    assert (source/'COMPLETE').exists() and lower<=state['completed_epochs']<args.num_train_epochs
    assert source.name == f"checkpoint-{state['completed_epochs']}"
    assert state['skipped_updates'] == 0
    assert state['global_optimizer_updates']==state['completed_epochs']*532
    if source.parent==out: assert state['run_id']==BRANCH and state['current_lr']==[5e-5]
    else: assert state['run_id']==MAIN and state['completed_epochs']==100 and state['current_lr']==[1e-4]

def restore(args,accelerator,optimizer,models,loader,dataset,progress):
    validate(args)
    source=Path(args.resume_from);initial=source.parent.name==MAIN
    assert progress.run_id==(MAIN if initial else BRANCH)
    # No RNG-consuming operations between load_state and this audit.
    saved_rng=torch.load(source/f'random_states_{accelerator.process_index}.pkl',map_location='cpu',weights_only=False)
    equal(random.getstate(),saved_rng['random_state']);equal(np.random.get_state(),saved_rng['numpy_random_seed']);equal(torch.get_rng_state(),saved_rng['torch_manual_seed'])
    equal(torch.cuda.get_rng_state_all(),saved_rng['torch_cuda_manual_seed'])
    for i,model in enumerate(models):
        path=source/('pytorch_model.bin' if i==0 else f'pytorch_model_{i}.bin')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        equal(saved,accelerator.unwrap_model(model).state_dict());del saved
    saved=torch.load(source/'optimizer.bin',map_location='cpu',weights_only=False)
    equal(saved,optimizer.state_dict())
    assert all(g['lr']==(1e-4 if initial else 5e-5) for g in optimizer.param_groups)
    before=copy.deepcopy(saved)
    for g in optimizer.param_groups:g['lr']=5e-5
    expected=copy.deepcopy(saved)
    for g in expected['param_groups']:g['lr']=5e-5
    equal(expected,optimizer.state_dict())
    equal(loader.generator.get_state(), progress.loader_generator_state)
    # Predict the first resumed epoch using an isolated sampler copy, without loading data.
    loader.set_epoch(progress.completed_epochs)
    sampler=copy.deepcopy(loader.batch_sampler, {id(dataset): dataset})
    batches=list(sampler)
    ids=np.asarray(dataset.base_dataset.split_presented_imgs)[np.asarray(batches)]
    gathered=accelerator.gather(torch.tensor(ids,device=accelerator.device)).cpu().numpy()
    ordered=gathered.reshape(accelerator.num_processes,len(batches),-1).transpose(1,0,2).reshape(-1)
    digest=hashlib.sha256(ordered.astype(np.int64).tobytes()).hexdigest()
    matches_h=None
    if args.num_train_epochs==110:
        reference={x['epoch']:x for x in map(json.loads,(ROOT/'runs'/MAIN/'epochs.jsonl').read_text().splitlines())}
        assert digest==reference[progress.completed_epochs+1]['image_order_sha256'], 'Branch sampler order differs from H'
        matches_h=True
    equal(torch.get_rng_state(),saved_rng['torch_manual_seed']);equal(torch.cuda.get_rng_state_all(),saved_rng['torch_cuda_manual_seed'])
    provenance=dict(source_run=MAIN,source_checkpoint=str(ROOT/'runs'/MAIN/'checkpoint-100'),source_commit=json.loads((ROOT/'runs'/MAIN/'exports/epoch-100/metadata.json').read_text())['code_commit'],branch_run_id=BRANCH,lr_change_location='after accelerator.load_state; before training_loop',resume_source=str(source),rank=accelerator.process_index,completed_epochs=progress.completed_epochs,global_optimizer_updates=progress.global_optimizer_updates,model_exact=True,adamw_before_exact=True,adamw_after_only_lr_changed=True,adamw_state_entries=len(saved['state']),optimizer_lr=[g['lr'] for g in optimizer.param_groups],rng_exact=True,loader_restored=True,next_epoch_order_sha256=digest,order_matches_H=matches_h, continuation_target=args.num_train_epochs, adamw_lr_unchanged=not initial)
    Path(args.output_dir,f'branch-resume-{progress.completed_epochs}-rank{accelerator.process_index}.json').write_text(json.dumps(provenance,indent=2))
    progress.run_id=BRANCH;progress.current_lr=[5e-5];progress.branch_provenance=provenance
    accelerator.wait_for_everyone()
    print('BRANCH_RESUME_AUDIT '+json.dumps(provenance),flush=True)
