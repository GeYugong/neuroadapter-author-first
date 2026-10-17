"""Subject 2 scoped launcher: original objective with telemetry and guarded retention."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import train_brain_adapter as t
from brain_adapter.dataset import nsd_topk_parcel_dataset
from types import SimpleNamespace


def write(path, obj):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def health_decision(epochs, probes):
    """Engineering thresholds, not a statistical significance test."""
    if len(epochs) < 10 or len(probes) < 10:
        return 'warmup'
    past_train = [np.mean([x['mean_loss'] for x in epochs[i:i+3]]) for i in range(len(epochs)-5)]
    past_val = [np.mean([x['loss'] for x in probes[i:i+3]]) for i in range(len(probes)-5)]
    now_train = np.mean([x['mean_loss'] for x in epochs[-3:]])
    now_val = np.mean([x['loss'] for x in probes[-3:]])
    train_bad = now_train > min(past_train) * 1.20 + 0.005
    val_bad = now_val > min(past_val) * 1.10 + 0.002
    if train_bad and val_bad:
        return 'stop_and_review_sustained_regression'
    if train_bad or val_bad:
        return 'review_trend_without_automatic_rollback'
    return 'healthy'


def install(args, smoke):
    out = Path(args.output_dir)
    state = dict(loss=[], kept=[], timesteps=[0]*4)
    original_setup = t.setup_accelerator
    original_models = t.setup_models_and_optimizer
    original_batch = t.process_training_batch
    original_loop = t.training_loop
    original_export = t.export_inference
    original_dropout = t.apply_fmri_token_dropout
    original_weight = t.min_snr_loss_weights

    def setup(a):
        acc = original_setup(a)
        assert acc.num_processes == 2 and acc.mixed_precision == 'bf16'
        state['acc'] = acc
        clip = acc.clip_grad_norm_
        def record_clip(params, max_norm, *pa, **kw):
            state['group_grad'] = {k: math.sqrt(sum(float(p.grad.detach().float().norm().square())
                                for p in group if p.grad is not None)) for k, group in state['groups'].items()}
            norm = clip(params, max_norm, *pa, **kw)
            state['grad_norm'] = float(norm)
            return norm
        acc.clip_grad_norm_ = record_clip
        return acc

    def models(*pa, **kw):
        result = original_models(*pa, **kw)
        adapter, guidance, opt, raw, _ = result
        acc = state['acc']
        state['models'] = (adapter, guidance)
        state['groups'] = dict(mapper=list(guidance.parameters()), projection=list(raw.image_proj_model.parameters()),
                               ip_kv=list(raw.adapter_modules.parameters()))
        step = opt.step
        def update(*a, **k):
            if not acc.sync_gradients:
                return step(*a, **k)
            number = args.progress.global_optimizer_updates + 1
            snapshot = number == 1
            before = {n: [p.detach().clone() for p in group] for n, group in state['groups'].items()} if snapshot else None
            ret = step(*a, **k)
            ratios = {}
            if snapshot:
                for n, group in state['groups'].items():
                    delta = sum(float((p.detach() - old).float().norm().square()) for p, old in zip(group, before[n]))
                    scale = sum(float(old.float().norm().square()) for old in before[n])
                    ratios[n] = math.sqrt(delta / max(scale, 1e-30))
                assert all(v > 0 and math.isfinite(v) for v in ratios.values()), ratios
                assert all(v > 0 and math.isfinite(v) for v in state['group_grad'].values())
            assert not acc.optimizer_step_was_skipped, 'Unexpected skipped BF16 optimizer update'
            record = dict(update=number, rank=acc.process_index, loss=float(np.mean(state['loss'])),
                          microbatches=len(state['loss']), grad_norm=state['grad_norm'], group_grad_norm=state['group_grad'],
                          update_to_weight_ratio=ratios, mean_kept_tokens=float(np.mean(state['kept'])),
                          timestep_counts=state['timesteps'], skipped=False, lr=opt.param_groups[0]['lr'],
                          peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3, unix_time=time.time())
            assert math.isfinite(record['loss']) and math.isfinite(record['grad_norm'])
            with (out / f'updates-rank{acc.process_index}.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if snapshot or number % 100 == 0:
                print('SUBJECT02_UPDATE ' + json.dumps(record), flush=True)
            state.update(loss=[], kept=[], timesteps=[0]*4)
            return ret
        opt.step = update
        return result

    def dropout(tokens):
        rng = torch.get_rng_state()
        result = original_dropout(tokens)
        gen = torch.Generator().set_state(rng)
        b, n, _ = tokens.shape
        probability = torch.rand(b, 1, generator=gen)
        mask = torch.rand(b, n, generator=gen) <= probability
        if not state.get('diagnostic'):
            state['kept'].extend(mask.sum(1).tolist())
        return result

    def weight(ts, *pa, **kw):
        if not state.get('diagnostic'):
            for v in ts.detach().cpu().tolist():
                state['timesteps'][min(int(v)//250, 3)] += 1
        return original_weight(ts, *pa, **kw)

    def batch(*pa, **kw):
        loss = original_batch(*pa, **kw)
        state['loss'].append(float(loss.detach()))
        return loss

    def probe(epoch, full=False):
        acc = state['acc']
        adapter, guidance, vae, text, scheduler, dtype, dataset = state['probe_components']
        data_args = SimpleNamespace(**vars(dataset.args))
        validation = nsd_topk_parcel_dataset(data_args, 'val', topk=100,
                                             selected_parcel_idx=dataset.selected_parcel_idx)
        base = validation.base_dataset
        ids = np.loadtxt(ROOT / ('data/subject02/val_ids.txt' if full else 'data/subject02/probe_ids.txt'), dtype=np.int64)
        base.split_presented_imgs = ids
        base.img_to_runs = [np.where(base.img_order == i)[0] for i in ids]
        loader = torch.utils.data.DataLoader(validation, batch_size=2, shuffle=False, num_workers=0)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        mode_a, mode_g = adapter.training, guidance.training
        adapter.eval(); guidance.eval()
        state['diagnostic'] = True
        values = []
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.no_grad():
                random.seed(91742); np.random.seed(91742); torch.manual_seed(91742); torch.cuda.manual_seed_all(91742)
                for b in loader:
                    # Unwrap DDP to avoid broadcasting while rank 1 waits at the epoch barrier.
                    with acc.autocast():
                        loss = original_batch(b, vae, scheduler, text, acc.unwrap_model(guidance),
                                              acc.unwrap_model(adapter), dtype, acc, validation)
                    values.append(float(loss))
        finally:
            random.setstate(python_rng); np.random.set_state(numpy_rng)
            adapter.train(mode_a); guidance.train(mode_g)
            state['diagnostic'] = False
        assert np.isfinite(values).all()
        record = dict(epoch=epoch, loss=float(np.mean(values)), samples=len(ids), seed=91742,
                      protocol='Fixed validation32, original U(0,1) dropout and MinSNR epsilon MSE; RNG restored')
        if full:
            record['protocol'] = 'Fixed validation500, original U(0,1) dropout and MinSNR epsilon MSE; RNG restored'
            write(out / f'validation-loss-{epoch}.json', record)
        else:
            with (out / 'fixed-probe.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
        return record

    def loop(a, acc, adapter, guidance, loader, opt, scheduler, vae, text, noise, dtype, dataset):
        state['probe_components'] = (adapter, guidance, vae, text, noise, dtype, dataset)
        if a.resume_from:
            saved = torch.load(Path(a.resume_from) / f'random_states_{acc.process_index}.pkl', map_location='cpu', weights_only=False)
            assert random.getstate() == saved['random_state']
            actual_np = np.random.get_state(); expected_np = saved['numpy_random_seed']
            assert actual_np[0] == expected_np[0] and np.array_equal(actual_np[1], expected_np[1]) and actual_np[2:] == expected_np[2:]
            assert torch.equal(torch.get_rng_state(), saved['torch_manual_seed'])
            actual_cuda = torch.cuda.get_rng_state_all()
            assert len(actual_cuda) == len(saved['torch_cuda_manual_seed'])
            assert all(torch.equal(x, y) for x, y in zip(actual_cuda, saved['torch_cuda_manual_seed']))
            write(out / f'resume-rank{acc.process_index}-to-{a.num_train_epochs}.json',
                  dict(source=a.resume_from, completed_epochs=a.progress.completed_epochs, full_rng_restored=True,
                       loader_restored=torch.equal(loader.generator.get_state(), a.progress.loader_generator_state),
                       optimizer_states=len(opt.state_dict()['state'])))
        write(out / f'actual-runtime-rank{acc.process_index}.json',
              dict(rank=acc.process_index, mixed_precision=acc.mixed_precision, frozen_vae_dtype=str(next(vae.parameters()).dtype),
                   trainable_dtypes=sorted({str(p.dtype) for group in state['groups'].values() for p in group}),
                   effective_batch=a.train_batch_size * a.gradient_accumulation_steps * acc.num_processes,
                   train_samples=len(dataset), parcels=dataset.num_parcels, max_voxels=dataset.max_voxels,
                   microbatches_per_rank=len(loader), updates_per_epoch=math.ceil(len(loader)/a.gradient_accumulation_steps)))
        if not smoke and not a.resume_from and acc.is_main_process:
            probe(0)
        acc.wait_for_everyone()
        return original_loop(a, acc, adapter, guidance, loader, opt, scheduler, vae, text, noise, dtype, dataset)

    def export(*pa, **kw):
        if pa[-1] in (0, 300, 500):
            return original_export(*pa, **kw)

    def cleanup(directory, keep_last_n=2):
        assert Path(directory).resolve() == out.resolve() and out.resolve().is_relative_to(ROOT / 'runs')
        checkpoints = sorted(out.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
        assert all((p / 'COMPLETE').exists() for p in checkpoints)
        if smoke:
            return
        epoch = args.progress.completed_epochs
        probe(epoch)
        if epoch in (300, 500):
            probe(epoch, full=True)
        epochs = [json.loads(x) for x in (out/'epochs.jsonl').read_text().splitlines()]
        probes = [json.loads(x) for x in (out/'fixed-probe.jsonl').read_text().splitlines() if json.loads(x)['epoch'] > 0]
        decision = health_decision(epochs, probes)
        stable = min(probes, key=lambda x: (x['loss'], x['epoch']))['epoch']
        keep = {p.name for p in checkpoints[-2:]} | {'checkpoint-300','checkpoint-500',f'checkpoint-{stable}'}
        record = dict(epoch=epoch, decision=decision, stable_probe_epoch=stable, latest_probe=probes[-1],
                      recent_training_loss=[x['mean_loss'] for x in epochs[-5:]], time=time.time())
        write(out/'health.json', record)
        if decision.startswith('stop_') or (out/'STOP_AFTER_EPOCH').exists():
            write(out/'intervention-required.json', record)
            os.kill(os.getpid(), signal.SIGTERM)
            return
        for p in checkpoints:
            if p.name not in keep:
                assert p.resolve().parent == out.resolve() and not p.is_symlink()
                shutil.rmtree(p)
        if shutil.disk_usage(ROOT).free < 25 * 1024**3:
            write(out/'intervention-required.json', dict(reason='disk_reserve_below_25GiB', epoch=epoch))
            os.kill(os.getpid(), signal.SIGTERM)

    t.setup_accelerator = setup
    t.setup_models_and_optimizer = models
    t.apply_fmri_token_dropout = dropout
    t.min_snr_loss_weights = weight
    t.process_training_batch = batch
    t.training_loop = loop
    t.export_inference = export
    t.cleanup_checkpoints = cleanup


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--config', required=True)
    cli.add_argument('--target', type=int, required=True)
    cli.add_argument('--output', required=True)
    cli.add_argument('--resume')
    cli.add_argument('--smoke', action='store_true')
    p = cli.parse_args()
    config = json.loads(Path(p.config).read_text())
    tokens = []
    for key, value in config.items():
        tokens.extend(['--'+key, str(value)])
    args = t.create_argument_parser().parse_args(tokens)
    args.output_dir = str((ROOT/p.output).resolve())
    assert Path(args.output_dir).is_relative_to(ROOT/'runs/subject02-author-bf16-seed42')
    args.num_train_epochs = p.target
    assert 1 <= p.target <= 500 and args.subject_id == 2 and args.seed == 42
    assert args.learning_rate == 1e-4 and args.weight_decay == 1e-6 and args.mixed_precision == 'bf16'
    assert args.train_batch_size * args.gradient_accumulation_steps * 2 == 32
    args.resume_from = str((ROOT/p.resume).resolve()) if p.resume else None
    if p.smoke:
        args.run_id = 'subject02-isolated-preflight'
        args.train_ids = str(ROOT/'data/subject02/smoke_ids.txt')
        args.audit = True
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    args.time = args.run_id
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    install(args, p.smoke)
    t.main(args)


if __name__ == '__main__':
    main()
