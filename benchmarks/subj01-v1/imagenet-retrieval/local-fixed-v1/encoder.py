"""Original author modules, shared frozen DINO, exact masked linear readout.

No learned operation or ensemble member is replaced. Full original forward is
compared on fixed candidates before using the memory-efficient readout.
"""
import os
import sys
sys.dont_write_bytecode = True
from common import *
os.environ['TORCH_HOME'] = str(ROOT / 'runtime/torch')
os.environ['MPLCONFIGDIR'] = str(ASSETS / 'mpl')
import gc
from types import SimpleNamespace
import numpy as np
import torch


class HalfEncoder:
    def __init__(self, hemi, device, check_batch):
        from importlib import import_module
        sys.path.insert(0, str(ENC))
        brain_encoder = import_module('models.brain_encoder').brain_encoder
        self.nested = import_module('utils.utils').nested_tensor_from_tensor_list
        self.device, self.hemi = device, hemi
        self.mask = torch.from_numpy(np.load(MASK)[hemi]).to(device)
        self.members, self.audit = [], []
        hub = torch.hub.load
        backbone_state = torch.load(ROOT / 'runtime/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth', map_location='cpu', weights_only=True)
        self.backbone = hub(str(DINO), 'dinov2_vitb14', source='local', pretrained=False)
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.backbone.eval().requires_grad_(False)
        def shared_hub(repo, name, *args, **kwargs):
            assert name == 'dinov2_vitb14' and kwargs == {'source': 'local'}
            return self.backbone
        torch.hub.load = shared_hub
        ps = torch.load(ENC / f'parcels/schaefer/{hemi}_labels_s01.pt', map_location='cpu', weights_only=True)
        dataset = SimpleNamespace(num_parcels=501, parcels=ps, num_hemi_voxels=163842,
                                  valid_voxel_mask=torch.ones(163842, dtype=torch.bool))
        parcel_ids = np.empty(163842, dtype=np.int64)
        for pi, verts in enumerate(ps):
            parcel_ids[verts.numpy()] = pi
        vertices = np.flatnonzero(self.mask.cpu().numpy())
        active = np.unique(parcel_ids[vertices])
        max_len = max(np.count_nonzero(parcel_ids[vertices] == p) for p in active)
        destinations = np.full((len(active), max_len), -1, dtype=np.int64)
        readout_vertices = np.zeros_like(destinations)
        for j, p in enumerate(active):
            outidx = np.flatnonzero(parcel_ids[vertices] == p)
            destinations[j, :len(outidx)] = outidx
            readout_vertices[j, :len(outidx)] = vertices[outidx]
        valid = destinations >= 0
        self.active = torch.from_numpy(active).to(device)
        self.destination = torch.from_numpy(destinations[valid]).to(device)
        self.valid = torch.from_numpy(valid).to(device)
        corr = []
        try:
            for layer in [1, 3, 5, 7]:
                for run in [1, 2]:
                    folder = WEIGHTS / f'enc_{layer}/run_{run}/{hemi}'
                    checkpoint = torch.load(folder / 'checkpoint_nonavg.pth', map_location='cpu', weights_only=False)
                    cfg = checkpoint['args']; cfg.device = device
                    assert cfg.enc_output_layer == layer and cfg.run == run and cfg.hemi == hemi
                    model = brain_encoder(cfg, dataset)
                    state = {'backbone_model.0.backbone.' + k: v for k, v in backbone_state.items()}
                    state.update(checkpoint['model'])
                    model.load_state_dict(state, strict=True)
                    model.to(device).eval().requires_grad_(False)
                    model.parcel_mask = model.parcel_mask.to(device)
                    model.valid_voxel_mask = model.valid_voxel_mask.to(device)
                    ix = torch.from_numpy(readout_vertices).to(device)
                    w = model.embed[0].weight[ix].contiguous()
                    bias = model.embed[0].bias[ix].contiguous()
                    with torch.inference_mode():
                        ref = model(check_batch[:2].to(device))
                        assert torch.isfinite(ref['pred']).all()
                        optimized = self.readout(ref['output_tokens'], w, bias)
                        expected = ref['pred'][:, self.mask]
                        error = float((expected - optimized).abs().max())
                        assert torch.allclose(expected, optimized, atol=2e-4, rtol=2e-4), error
                    self.audit.append(dict(hemi=hemi, layer=layer, run=run, full_state_strict=True,
                        original_full_forward_finite=True, masked_readout_max_abs=error,
                        active_parcels=len(active), padded_vertices=int(destinations.size),
                        selected_vertices=len(vertices)))
                    # Keep original transformer, query embeddings, and backbone.
                    # Omitted readout rows cannot affect other vertices (linear
                    # readout followed by the author's one-hot parcel mask).
                    model.embed = torch.nn.Identity()
                    del model.parcel_mask
                    self.members.append((model, w, bias))
                    corr.append(np.load(folder / f'{hemi}_val_corr_nonavg.npy'))
                    del checkpoint, state, ref, expected, optimized
                    gc.collect(); torch.cuda.empty_cache()
        finally:
            torch.hub.load = hub
        self.weights = torch.softmax(20 * torch.from_numpy(np.stack(corr)), dim=0)[:, self.mask.cpu()].to(device)
        assert torch.isfinite(self.weights).all()
        self.original_intermediate = self.backbone.get_intermediate_layers
        self.cache = None
        def cached(*args, **kwargs):
            if self.cache is None:
                self.cache = self.original_intermediate(*args, **kwargs)
            return self.cache
        self.backbone.get_intermediate_layers = cached
        del backbone_state

    def readout(self, tokens, weights, bias):
        values = torch.bmm(tokens[:, self.active].transpose(0, 1), weights.transpose(1, 2))
        values = (values + bias[:, None]).permute(1, 0, 2)
        out = torch.empty((len(tokens), int(self.mask.sum())), dtype=torch.float32, device=self.device)
        out[:, self.destination] = values[:, self.valid]
        return out

    @torch.inference_mode()
    def predict(self, batch, use_cache=True):
        samples = self.nested(batch.to(self.device))
        self.cache = None
        parts = []
        for model, w, bias in self.members:
            if not use_cache:
                self.cache = None
            features, pos = model.backbone_model(samples)
            src, mask = features[-1].decompose()
            hs = model.transformer(src, mask, model.query_embed.weight, pos[-1], model.return_interm)
            prediction = self.readout(hs[-1], w, bias)
            if not torch.isfinite(prediction).all():
                raise ValueError(f'Nonfinite model output: {self.hemi}')
            parts.append(prediction)
        result = (torch.stack(parts) * self.weights[:, None, :]).sum(dim=0)
        assert torch.isfinite(result).all()
        self.cache = None
        return result.cpu().numpy()
