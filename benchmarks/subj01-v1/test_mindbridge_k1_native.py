"""CPU differential control-flow test; not a real model or benchmark result."""
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from mindbridge_k1_native import compile_native


class Brain:
    def __call__(self, voxel):
        image = torch.arange(257 * 768, dtype=torch.float32).reshape(1, -1) / 10000 + voxel.mean()
        text = torch.arange(77 * 768, dtype=torch.float32).reshape(1, -1) / 20000 + voxel[:, :4096].mean()
        return image, text


class UNet:
    dtype = torch.float32
    in_channels = 4
    config = SimpleNamespace(sample_size=64)

    def __init__(self):
        self.conditions = []

    def to(self, device):
        return self

    def __call__(self, latent, timestep, encoder_hidden_states):
        self.conditions.append(encoder_hidden_states.clone())
        image = encoder_hidden_states[:, 77:].mean(dim=(1, 2))
        text = encoder_hidden_states[:, :77].mean(dim=(1, 2))
        return SimpleNamespace(sample=latent * .1 + (image + text * 2)[:, None, None, None])


class VAE:
    config = SimpleNamespace(block_out_channels=[1, 1, 1, 1])

    def to(self, device):
        return self

    def decode(self, latent):
        return SimpleNamespace(sample=latent[:, :3].tanh())


class Scheduler:
    init_noise_sigma = 1.0

    def set_timesteps(self, num_inference_steps, device):
        self.timesteps = list(range(num_inference_steps - 1, -1, -1))

    def scale_model_input(self, latent, timestep):
        return latent

    def step(self, predicted, timestep, latent):
        return SimpleNamespace(prev_sample=latent - predicted * .01)


class UnusedClip:
    def to(self, device):
        return self

    def embed_image(self, *args, **kwargs):
        raise AssertionError('GT selector must never execute')


def main():
    torch.set_num_threads(2)
    k1, reference, audit = compile_native(torch)
    assert list(inspect.signature(k1).parameters) == ['voxel', 'voxel2clip', 'unet', 'vae', 'noise_scheduler', 'device']
    voxel = torch.linspace(-1, 2, 8192).reshape(1, -1)
    unet = UNet()
    state = torch.get_rng_state().clone()
    actual = k1(voxel, Brain(), unet, VAE(), Scheduler(), device='cpu')
    assert torch.equal(state, torch.get_rng_state()), 'Native per-image private generator must leave global CPU RNG unchanged'
    assert len(unet.conditions) == 20
    condition = unet.conditions[0]
    assert condition.shape == (2, 334, 768)
    assert condition[0].count_nonzero() == 0
    assert condition[1, :77].count_nonzero() > 0 and condition[1, 77:].count_nonzero() > 0
    # Prefix of the author's unchanged function, before all GT selection. Synthetic
    # placeholder image only: no real GT is read, no CLIP image encoding is allowed.
    references = []
    for synthetic in (torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8)):
        ref = reference(synthetic, voxel, Brain(), UnusedClip(), UNet(), VAE(), Scheduler(),
                        img_lowlevel=None, num_inference_steps=20, recons_per_sample=1,
                        guidance_scale=5., img2img_strength=1., seed=42, plotting=False,
                        verbose=False, n_samples_save=1, device='cpu', mem_efficient=False)
        assert torch.equal(actual, ref)
        references.append(True)
    second = k1(voxel, Brain(), UNet(), VAE(), Scheduler(), device='cpu')
    assert torch.equal(actual, second), 'Per-image seed changed across invocations'
    assert torch.isfinite(actual).all() and actual.shape == (1, 1, 3, 64, 64)
    assert not torch.cuda.is_initialized()
    result = dict(audit, cpu_stub_differential_equal=references, both_predicted_conditions_present=True,
                  native_global_rng_unchanged=True, repeat_call_identical=True,
                  cuda_initialized=False, real_model_inference=False, benchmark_candidates=0,
                  limitation='Synthetic modules verify adapter control flow only; full600 weights, actual GPU identity and benchmark remain pending')
    target = Path(__file__).resolve().parent / 'mindbridge-k1-adapter-cpu-test.json'
    assert not target.exists(), 'Keep the previous test receipt; inspect before another attempt'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
