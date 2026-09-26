#!/usr/bin/env python
"""Thin MindEye1 K1 adapter; imports the pinned author's model and reconstruction.

--brain-inputs: NPZ with nsd_id0 int64[N], voxels float32[N,15724].
voxels must be the original test nsdgeneral repeat mean, without new normalization.
--ids: one zero-based NSD global ID per line; no image/GT input is accepted.
Run in the isolated MindEye environment. Upstream and assets are explicit paths.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--brain-inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--high-checkpoint", type=Path, required=True)
    p.add_argument("--low-checkpoint", type=Path, required=True)
    p.add_argument("--vd", type=Path, required=True)
    p.add_argument("--clip-vision", type=Path, required=True,
                   help="Pinned local openai/clip-vit-large-patch14 snapshot, config plus weights")
    p.add_argument("--openai-clip", type=Path, required=True,
                   help="Local official OpenAI ViT-L-14.pt checkpoint")
    args = p.parse_args()
    import numpy as np
    import torch
    from torchvision.transforms.functional import to_pil_image

    ids = [int(x) for x in args.ids.read_text().split()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("IDs must be nonempty and unique")
    with np.load(args.brain_inputs, allow_pickle=False) as data:
        source_ids = data["nsd_id0"].astype(np.int64)
        voxels = data["voxels"]
    if voxels.dtype != np.float32 or voxels.shape != (len(source_ids), 15724):
        raise ValueError("Expected float32 voxels[N,15724]")
    if len(set(source_ids.tolist())) != len(source_ids) or not np.isfinite(voxels).all():
        raise ValueError("Duplicate source IDs or nonfinite brain input")
    lookup = {int(i): n for n, i in enumerate(source_ids)}
    if not set(ids).issubset(lookup):
        raise ValueError("Requested IDs missing in brain inputs")
    for path in (args.high_checkpoint, args.low_checkpoint, args.openai_clip):
        if not path.is_file():
            raise FileNotFoundError(path)
    upstream = args.upstream.resolve()
    utils_path = upstream / "src" / "utils.py"
    utils_source = utils_path.read_text()
    retrieval_import = "from clip_retrieval.clip_client import ClipClient"
    if utils_source.count(retrieval_import) != 1:
        raise RuntimeError("Expected exactly one author retrieval import")
    retrieval_disabled = ("class ClipClient:\n"
                          "    def __init__(self, *args, **kwargs):\n"
                          "        raise RuntimeError('LAION retrieval disabled in K1 benchmark')")
    adapted_utils = utils_source.replace(retrieval_import, retrieval_disabled)
    vision_files = [x for x in args.clip_vision.rglob("*") if x.is_file()
                    and x.name in ("config.json", "pytorch_model.bin", "model.safetensors")]
    if not (args.clip_vision / "config.json").is_file() or not any(
            x.name in ("pytorch_model.bin", "model.safetensors") for x in vision_files):
        raise FileNotFoundError("CLIP vision local config and weights required")
    code_sha = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    protocol = dict(method="MindEye1", subject=1, K=1, selection="none", seed=42,
                    seed_policy="author fixed seed42, fresh generator for every image",
                    batch_size=1, num_inference_steps=20, guidance_scale=3.5,
                    img2img_strength=0.85, timesteps_prior=100, text_image_ratio=0.0,
                    high_sha256=sha(args.high_checkpoint), low_sha256=sha(args.low_checkpoint),
                    brain_inputs_sha256=sha(args.brain_inputs), code_sha=code_sha,
                    source_files={name: sha(upstream / "src" / name)
                                  for name in ("models.py", "utils.py", "Reconstructions.py")},
                    adapter_sha256=sha(Path(__file__)), vd_path=str(args.vd.resolve()),
                    clip_vision_files={str(x.relative_to(args.clip_vision)): sha(x) for x in sorted(vision_files)},
                    openai_clip_sha256=sha(args.openai_clip),
                    runtime_import_patch=dict(original_line=retrieval_import,
                        replacement=retrieval_disabled,
                        adapted_utils_sha256=hashlib.sha256(adapted_utils.encode()).hexdigest()),
                    result_source="local_inference_official_weights")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise RuntimeError("Existing output protocol differs; refusing overwrite")
    if not lock.exists():
        write_json(lock, protocol)
    todo = []
    for image_id in ids:
        record = args.output / f"{image_id}.json"
        png = args.output / f"{image_id}.png"
        if record.exists():
            prior = json.loads(record.read_text())
            if prior["nsd_id0"] != image_id or not png.exists() or sha(png) != prior["prediction_sha256"]:
                raise RuntimeError(f"Invalid completed output {image_id}")
        elif png.exists():
            raise RuntimeError(f"Uncommitted PNG {image_id}; inspect before recovery")
        else:
            todo.append(image_id)
    if not todo:
        print("All requested IDs already verified", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for actual inference")
    device = torch.device("cuda:0")
    sys.path.insert(0, str(upstream / "src"))
    utils = types.ModuleType("utils")
    utils.__file__ = str(utils_path)
    sys.modules["utils"] = utils
    exec(compile(adapted_utils, str(utils_path), "exec"), utils.__dict__)
    from models import (Clipper, BrainNetwork, BrainDiffusionPrior,
                        Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork)
    from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    utils.device = device
    utils.seed_everything(42)

    def load_model(model, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"]
        result = model.load_state_dict(state, strict=True)
        audit = dict(path=str(path.resolve()), epoch=checkpoint.get("epoch"),
                     missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys)
        del checkpoint, state
        gc.collect()
        return model.eval().requires_grad_(False), audit

    low, low_audit = load_model(Voxel2StableDiffusionModel(in_dim=15724, use_cont=True), args.low_checkpoint)
    brain = BrainNetwork(in_dim=15724, out_dim=257 * 768)
    # Official checkpoint retains a training temperature parameter; author forward never reads it.
    brain.register_parameter("temp", torch.nn.Parameter(torch.tensor(.006), requires_grad=False))
    net = VersatileDiffusionPriorNetwork(dim=768, depth=6, dim_head=64, heads=12,
                                       causal=False, learned_query_mode="pos_emb")
    prior = BrainDiffusionPrior(net=net, image_embed_dim=768,
                               condition_on_text_encodings=False, timesteps=100,
                               cond_drop_prob=0.2, image_embed_scale=None, voxel2clip=brain)
    prior, high_audit = load_model(prior, args.high_checkpoint)
    # Fail closed if the author's broad exception attempts a no-generator fallback.
    original_loop = prior.p_sample_loop
    def guarded_loop(self, *a, **kw):
        if kw.get("generator") is None:
            raise RuntimeError("No-generator prior fallback prohibited")
        return original_loop(*a, **kw)
    prior.p_sample_loop = types.MethodType(guarded_loop, prior)
    vd = VersatileDiffusionDualGuidedPipeline.from_pretrained(
        str(args.vd), torch_dtype=torch.float16, local_files_only=True)
    dual_count = 0
    for module in vd.image_unet.modules():
        if isinstance(module, DualTransformer2DModel):
            dual_count += 1
            module.mix_ratio = 0.0
            module.condition_lengths[0], module.condition_lengths[1] = 77, 257
            module.transformer_index_for_condition[0] = 1
            module.transformer_index_for_condition[1] = 0
    if not dual_count:
        raise RuntimeError("VD dual attention not constructed; text_unet asset may be missing")
    vd.remove_unused_weights()
    vd.scheduler = UniPCMultistepScheduler.from_pretrained(str(args.vd), subfolder="scheduler", local_files_only=True)
    vd.image_unet.eval().requires_grad_(False).to(device)
    vd.vae.eval().requires_grad_(False).to(device)
    low.to(device)
    prior.to(device)
    import clip
    from transformers import CLIPVisionModelWithProjection
    original_clip_load = clip.load
    original_hf_descriptor = CLIPVisionModelWithProjection.__dict__.get("from_pretrained")
    original_hf_load = CLIPVisionModelWithProjection.from_pretrained
    def local_clip_load(name, *a, **kw):
        if name != "ViT-L/14":
            raise RuntimeError("Unexpected OpenAI CLIP request")
        return original_clip_load(str(args.openai_clip), *a, **kw)
    def local_vision_load(cls, name, *a, **kw):
        if name != "openai/clip-vit-large-patch14":
            raise RuntimeError("Unexpected HF CLIP request")
        kw["local_files_only"] = True
        return original_hf_load(str(args.clip_vision), *a, **kw)
    clip.load = local_clip_load
    CLIPVisionModelWithProjection.from_pretrained = classmethod(local_vision_load)
    try:
        clipper = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
    finally:
        clip.load = original_clip_load
        if original_hf_descriptor is None:
            delattr(CLIPVisionModelWithProjection, "from_pretrained")
        else:
            CLIPVisionModelWithProjection.from_pretrained = original_hf_descriptor
    write_json(args.output / "load_audit.json", dict(low=low_audit, high=high_audit,
               dual_attention_modules=dual_count, low_level_loaded=True,
               torch_version=torch.__version__, cuda_name=torch.cuda.get_device_name(device)))
    with torch.no_grad():
        for image_id in todo:
            voxel = torch.from_numpy(voxels[lookup[image_id]:lookup[image_id]+1]).to(device)
            ae_preds = low(voxel.float())
            blurry = vd.vae.decode(ae_preds.half() / 0.18215).sample / 2 + 0.5
            # Placeholder is used only by the author's disabled plotting path.
            placeholder = torch.zeros((1, 3, 512, 512))
            _, recons, picks, _ = utils.reconstruction(
                placeholder, voxel, clipper, vd.image_unet, vd.vae, vd.scheduler,
                voxel2clip_cls=None, diffusion_priors=[prior], text_token=None,
                img_lowlevel=blurry, num_inference_steps=20, n_samples_save=1,
                recons_per_sample=1, guidance_scale=3.5, img2img_strength=0.85,
                timesteps_prior=100, seed=42, retrieve=False, plotting=False,
                img_variations=False, verbose=False)
            if list(picks) != [0] or tuple(recons.shape) != (1, 1, 3, 512, 512):
                raise RuntimeError("Unexpected K1 output")
            result = recons[0, 0].float().cpu()
            if not torch.isfinite(result).all():
                raise RuntimeError("Nonfinite output")
            png = args.output / f"{image_id}.png"
            tmp = args.output / f"{image_id}.png.tmp"
            to_pil_image(result.clamp(0, 1)).save(tmp, format="PNG")
            os.replace(tmp, png)
            write_json(args.output / f"{image_id}.json", dict(nsd_id0=image_id,
                       candidate=0, seed=42, prediction_path=png.name,
                       prediction_sha256=sha(png), protocol_path="protocol.json",
                       low_level_used=True, cuda_peak_allocated=torch.cuda.max_memory_allocated(device)))
            print(json.dumps(dict(nsd_id0=image_id, status="complete")), flush=True)
    write_json(args.output / "last_request_complete.json", dict(ids=ids, n=len(ids)))


if __name__ == "__main__":
    main()
