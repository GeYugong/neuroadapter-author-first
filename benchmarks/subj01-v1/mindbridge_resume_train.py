"""Resume the interrupted epoch420 run; author math stays in the upstream module."""
import ast
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import runpy
import sys
import textwrap

import numpy as np
import torch
import torch.distributed as dist

from mindbridge_local_assets import configure

B = Path(__file__).resolve().parent
SOURCE = B / 'upstream/mindbridge/src'
SNAPSHOT = B / 'processed/mindbridge-recovery-epoch420'
configure()
os.chdir(SOURCE)
sys.path.insert(0, str(SOURCE))
import trainer as author


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def rng():
    return dict(random_state=random.getstate(), numpy_random_seed=np.random.get_state(),
                torch_manual_seed=torch.get_rng_state(),
                torch_cuda_manual_seed=torch.cuda.get_rng_state_all())


def restore_rng(states):
    random.setstate(states['random_state'])
    np.random.set_state(states['numpy_random_seed'])
    torch.set_rng_state(states['torch_manual_seed'])
    torch.cuda.set_rng_state_all(states['torch_cuda_manual_seed'])


def broadcast(value):
    box = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def metadata(self, tag, epoch):
    path = Path(self.outdir) / tag
    rank = self.accelerator.process_index
    torch.save(self.train_dl.synchronized_generator.get_state(), path / f'loader_generator_{rank}.pt')
    if rank == 0:
        (path / 'trainer-recovery.json').write_text(json.dumps(dict(
            completed_epoch_zero_based=epoch, next_epoch=epoch + 1,
            best_sim=self.best_sim, best_epoch=self.best_epoch,
            selection='Original rank0 native validation similarity; shared decision across ranks'), indent=2) + '\n')
    self.accelerator.wait_for_everyone()


def synchronized_save(self, epoch):
    current = (self.val_sims_image + self.val_sims_text) / (self.val_i + 1) if hasattr(self, 'val_i') else 0
    decision, best_sim, best_epoch = broadcast((current > self.best_sim,
        current if current > self.best_sim else self.best_sim,
        epoch if current > self.best_sim else self.best_epoch))
    self.best_sim, self.best_epoch = best_sim, best_epoch
    self.save_ckpt('last', epoch)
    metadata(self, 'last', epoch)
    if decision:
        self.save_ckpt('best', epoch)
        metadata(self, 'best', epoch)
    if self.accelerator.is_main_process:
        print('SYNCHRONIZED_SAVE', epoch, decision, best_sim, best_epoch, flush=True)


def audited_resume(self):
    rank = self.accelerator.process_index
    self.accelerator.load_state(str(SNAPSHOT / 'last'))
    engine = self.voxel2clip
    assert engine.global_steps == 36206 and engine.skipped_steps == 0
    weights = torch.load(SNAPSHOT / 'last.pth', map_location='cpu')
    assert weights['epoch'] == 420
    equal(weights['model_state_dict'], self.accelerator.unwrap_model(engine).state_dict())
    best = torch.load(SNAPSHOT / 'best.pth', map_location='cpu')
    assert best['epoch'] == 420
    equal(weights['model_state_dict'], best['model_state_dict'])
    del weights, best
    saved = torch.load(SNAPSHOT / f'last/pytorch_model/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt', map_location='cpu')['optimizer_state_dict']
    live = engine.optimizer.state_dict()
    for k in ('base_optimizer_state', 'single_partition_of_fp32_groups'):
        equal(saved[k], live[k])
    del saved, live
    scheduler = torch.load(SNAPSHOT / 'last/scheduler.bin', map_location='cpu')
    live_scheduler = self.lr_scheduler.state_dict()
    for k in scheduler:
        if k != 'anneal_func':  # Bound method contains an optimizer object, not numeric scheduler state.
            equal(scheduler[k], live_scheduler[k])
    assert scheduler['last_epoch'] == 51600
    lr = scheduler['_last_lr']
    del scheduler, live_scheduler
    saved_rng = torch.load(SNAPSHOT / f'last/random_states_{rank}.pkl', map_location='cpu')
    equal({k: saved_rng[k] for k in rng()}, rng())
    self.epoch_start = 421  # Original resume() repeats the just-completed epoch; explicitly avoid that.

    # Accelerate0.19 creates a separate, default-seeded RandomSampler generator.
    # It is absent from original save_state. Replay ONLY index sampling, no data,
    # model, optimizer or global RNG, to recover its exact epoch-boundary state.
    generator = self.train_dl.synchronized_generator
    equal(generator.get_state(), torch.Generator().get_state())
    assert len(self.train_dl.dataset) == 8559 and len(self.train_dl) == 86
    for _ in range(self.epoch_start):
        assert sum(1 for _ in self.train_dl.batch_sampler) == 86
    generator_hash = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    assert len(set(all_gather(generator_hash))) == 1
    equal({k: saved_rng[k] for k in rng()}, rng())

    # Original best.pth is epoch420 but its threshold was not persisted.
    # Recompute its ORIGINAL deterministic native validation (not standard test),
    # then restore all saved RNG. No model or optimizer update occurs.
    self.val_losses = []
    self.val_voxel0 = self.val_image0 = None
    for name in ('sims_image', 'sims_text', 'fwd_percent_correct', 'bwd_percent_correct',
                 'loss_clip_image_sum', 'loss_clip_text_sum', 'loss_mse_image_sum',
                 'loss_mse_text_sum', 'loss_rec_sum', 'loss_cyc_sum'):
        setattr(self, 'val_' + name, 0.)
    self.eval_epoch(420)
    score = (self.val_sims_image + self.val_sims_text) / (self.val_i + 1)
    self.best_sim = self.recovered_best_sim = broadcast(score)
    self.best_epoch = self.recovered_best_epoch = 420
    assert np.isfinite(self.best_sim) and self.best_sim > 0
    restore_rng(saved_rng)
    equal({k: saved_rng[k] for k in rng()}, rng())
    assert engine.global_steps == 36206
    report = dict(rank=rank, resume_next_epoch_zero_based=421, completed_epochs=421,
                  global_steps=engine.global_steps, model_all_tensors_equal=True,
                  optimizer_moments_and_fp32_partitions_equal=True,
                  scheduler_numeric_state_equal=True, scheduler_last_epoch=51600,
                  learning_rates=lr, rng_all_equal=True,
                  loader_generator_sha256=generator_hash,
                  loader_recovery='421 exhausted original BatchSamplerShard iterations; indices only',
                  rank_native_validation_similarity=score, rank0_best_similarity=self.best_sim,
                  best_threshold_recovery='Reevaluated saved epoch420 on original validation; restored RNG afterwards',
                  limitation='Restart recovery verified at state level; no claim of bitwise future CUDA trajectory')
    (B / f'mindbridge-resume-audit-rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
    self.accelerator.wait_for_everyone()
    # Replace interrupted best state only after both ranks pass all audits.
    self.save_ckpt('best', 420)
    metadata(self, 'best', 420)
    restore_rng(saved_rng)
    self.accelerator.wait_for_everyone()
    if rank == 0:
        print('RESUME_AUDIT_PASSED_BOTH_RANKS next_epoch=421 global_steps=36206', flush=True)
    del saved_rng
    gc.collect()
    torch.cuda.empty_cache()


def all_gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


# Preserve the original train body except its two unconditional best-history resets.
tree = ast.parse(textwrap.dedent(inspect.getsource(author.Trainer.train)))
changed = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Attribute) and target.attr in ('best_sim', 'best_epoch'):
            assert isinstance(node.value, ast.Constant) and node.value.value == 0
            node.value = ast.parse(f'self.recovered_{target.attr}', mode='eval').body
            changed += 1
assert changed == 2
ast.fix_missing_locations(tree)
namespace = dict(vars(author))
exec(compile(tree, str(Path(__file__)), 'exec'), namespace)
author.Trainer.train = namespace['train']
author.Trainer.save = synchronized_save
author.Trainer.resume = audited_resume
assert '--resume' in sys.argv
runpy.run_path(str(SOURCE / 'main.py'), run_name='__main__')
