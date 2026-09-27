"""Read-only final checkpoint audit; writes a receipt only after all checks pass."""
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch

B = Path(__file__).resolve().parent
R = B.parents[1]
OUT = B / 'upstream/mindbridge/train_logs/benchmark-subj01-vanilla-seed42'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    receipt = B / 'mindbridge-training-complete.json'
    assert not receipt.exists(), 'Preserve completed audit'
    assert (B / 'logs/mindbridge-resume.exit').read_text().strip() == '0'
    for pid in (2443448, 2443472, 2443532, 2443533):
        assert not Path(f'/proc/{pid}').exists(), f'Training process {pid} still exists'
    meta = json.loads((OUT / 'last/trainer-recovery.json').read_text())
    assert meta['completed_epoch_zero_based'] == 599 and meta['next_epoch'] == 600
    checkpoint = torch.load(OUT / 'last.pth', map_location='cpu')
    assert checkpoint['epoch'] == 599
    exported = checkpoint['model_state_dict']
    assert sum(t.numel() for t in exported.values()) == 561283712
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in exported.values())
    engine = torch.load(OUT / 'last/pytorch_model/mp_rank_00_model_states.pt', map_location='cpu')
    assert engine['global_steps'] == 51600 and engine['skipped_steps'] == 0 and engine['dp_world_size'] == 2
    assert engine['module'].keys() == exported.keys()
    assert all(torch.equal(engine['module'][k], v) for k, v in exported.items())
    global_samples = engine['global_samples']
    del engine
    gc.collect()
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    recovered = get_fp32_state_dict_from_zero_checkpoint(str(OUT / 'last'))
    assert recovered.keys() == exported.keys()
    assert all(torch.equal(recovered[k], v) for k, v in exported.items()), 'ZeRO partition reconstruction differs from exported last'
    tensor_count = len(exported)
    del recovered, exported, checkpoint
    gc.collect()
    optimizer_steps = {}
    generators = []
    for rank in (0, 1):
        opt = torch.load(OUT / f'last/pytorch_model/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt', map_location='cpu')['optimizer_state_dict']
        assert opt['zero_stage'] == 2 and not opt['overflow']
        steps = [float(s['step']) for s in opt['base_optimizer_state']['state'].values()]
        # DS0.9.5 initialize_optimizer_states() calls optimizer.step() once on zero gradients.
        assert steps and all(s == 51601 for s in steps)
        optimizer_steps[str(rank)] = steps
        for state in opt['base_optimizer_state']['state'].values():
            assert all(torch.isfinite(state[k]).all() for k in ('exp_avg', 'exp_avg_sq'))
        assert all(torch.isfinite(v).all() for v in opt['single_partition_of_fp32_groups'])
        del opt
        rng = torch.load(OUT / f'last/random_states_{rank}.pkl', map_location='cpu')
        assert {'random_state', 'numpy_random_seed', 'torch_manual_seed', 'torch_cuda_manual_seed'} <= rng.keys()
        assert rng['torch_manual_seed'].dtype == torch.uint8 and len(rng['torch_cuda_manual_seed']) == 2
        loader = torch.load(OUT / f'last/loader_generator_{rank}.pt', map_location='cpu')
        assert loader.dtype == torch.uint8
        torch.Generator().set_state(loader)
        generators.append(hashlib.sha256(loader.numpy().tobytes()).hexdigest())
        del rng, loader
        gc.collect()
    assert len(set(generators)) == 1
    scheduler = torch.load(OUT / 'last/scheduler.bin', map_location='cpu')
    assert scheduler['last_epoch'] == 51600 and scheduler['total_steps'] == 51600
    assert all(abs(v - 2.4000055952305e-07) < 1e-15 for v in scheduler['_last_lr'])
    protected = json.loads((B / 'protected-before.json').read_text())
    assert all(sha(R / path) == value for path, value in protected.items())
    files = [OUT / 'last.pth'] + sorted(p for p in (OUT / 'last').rglob('*') if p.is_file())
    hashes = [dict(path=str(p.relative_to(R)), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    assert not torch.cuda.is_initialized()
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed_epochs=600,
        global_steps=51600, global_samples=global_samples, skipped_steps=0, resume_exit_code=0,
        checkpoint_path=str((OUT / 'last.pth').relative_to(R)), checkpoint_sha256=sha(OUT / 'last.pth'),
        full_state_verified=True, state_verification='CPU deserialize all final states; export equals model state and FP32 reconstruction from both ZeRO partitions; finite moments and readable RNG/loader/scheduler',
        tensor_count=tensor_count, parameters=561283712, export_equals_engine_module=True,
        export_equals_both_zero_partitions=True, optimizer_steps=optimizer_steps,
        optimizer_step_offset='DS0.9.5 initializes optimizer states with one zero-gradient step; engine records51600 actual training updates',
        scheduler_last_epoch=scheduler['last_epoch'], learning_rates=scheduler['_last_lr'],
        loader_generator_sha256=generators, trainer_metadata=meta, protected_files_unchanged=protected,
        files=hashes, cuda_initialized=False, inference_selection='Fixed complete600 last, not best',
        limitation='Completion audit is CPU state integrity/reconstruction, not a new distributed resume run or completed image benchmark')
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected_files_unchanged')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
