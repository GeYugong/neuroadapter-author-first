"""Audit public encoder assets on two ImageNet files; never load NSD GT or train.

This diagnostic does not authorize or perform the formal retrieval. Split IDs,
beta normalization provenance, and the retrieval producer's ROI definition still
need independent verification. Old source and weights are read-only.
"""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
os.environ["TORCH_HOME"] = str(ROOT / "runtime/torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime/imagenet-retrieval-assets/mpl")
import argparse
import gc
import hashlib
import json
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import nibabel as nib
import torch
from PIL import Image
from torchvision import transforms


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-source", type=Path, required=True)
    parser.add_argument("--dino-source", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--roi-reference", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    a = parser.parse_args()
    a.output = a.output.resolve()
    assert a.output.is_relative_to(ROOT) and not a.output.exists()
    gpu_processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader"], text=True).strip()
    assert not gpu_processes, "GPU already has compute processes; do not interfere"
    audit = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running", "training_updates": 0,
        "formal_candidates_scanned": 0, "nsd_gt_images_loaded": 0,
        "diagnostic_candidate_count": 2,
        "encoder_commit": subprocess.check_output([
            "git", "-C", str(a.encoder_source), "rev-parse", "HEAD"], text=True).strip(),
        "dino_commit": subprocess.check_output([
            "git", "-C", str(a.dino_source), "rev-parse", "HEAD"], text=True).strip(),
        "dino_weight_sha256": sha(a.dino_weight),
        "device": a.device, "dtype": "float32", "tf32": False,
        "members": [], "roi": {}, "images": [],
        "limits": ["No checkpoint training/validation ID provenance established",
                   "No beta normalization recipe established for these checkpoints",
                   "48 canonical ROI masks recovered; streams inclusion in the historical retrieval producer is unconfirmed",
                   "Forward checks on two candidates are not formal baseline metrics"],
    }
    partial = a.output.with_suffix(".partial.json")
    def save_partial():
        partial.write_text(json.dumps(audit, indent=2) + "\n")
    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        roi_source = a.encoder_source / "parcels/nsd_labels/labels_s01.pt"
        parcels = torch.load(roi_source, map_location="cpu", weights_only=True)
        raw_parcels = np.load(roi_source.with_suffix(".npy"), allow_pickle=True).item()["parcels"]
        assert len(parcels) == len(raw_parcels) == 48
        assert all(np.array_equal(x.numpy(), y) for x, y in zip(parcels, raw_parcels))
        audit["roi"]["public_asset_sha256"] = sha(roi_source)
        masks = {}
        for hi, hemi in enumerate(["lh", "rh"]):
            mapping_file = a.roi_reference / f"{hemi}.white-to-fsaverage.mgz"
            mapping = nib.load(mapping_file).get_fdata().squeeze().astype(np.int64) - 1
            source_masks, source_files = [], []
            for category in ["prf-visualrois", "floc-bodies", "floc-faces", "floc-places", "floc-words"]:
                file = a.roi_reference / f"{hemi}.{category}.mgz"
                labels = nib.load(file).get_fdata().squeeze()[mapping]
                ctab = (a.roi_reference / f"{category}.mgz.ctab").read_text().splitlines()
                ids = [int(line.split()[0]) for line in ctab if line.strip() and int(line.split()[0]) > 0]
                source_masks.extend(labels == v for v in ids)
                source_files.append({"path": str(file), "sha256": sha(file)})
            ref = np.stack(source_masks)
            recovered = np.zeros((24, 163842), dtype=bool)
            for j, ids in enumerate(parcels[hi * 24:(hi + 1) * 24]):
                local_ids = ids.numpy() - hi * 163842
                assert local_ids.size == 0 or (local_ids.min() >= 0 and local_ids.max() < 163842)
                recovered[j, local_ids] = True
            assert ref.shape == recovered.shape and np.array_equal(ref, recovered)
            masks[hemi] = recovered.any(axis=0)
            audit["roi"][hemi] = {
                "canonical_roi_count": 24, "individual_masks_exact_in_original_order": True,
                "union_vertices": int(masks[hemi].sum()),
                "union_sha256": hashlib.sha256(masks[hemi].tobytes()).hexdigest(),
                "mapping_sha256": sha(mapping_file), "reference_files": source_files,
            }
        asset_dir = ROOT / "runtime/imagenet-retrieval-assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        mask_file = asset_dir / "public-canonical-roi-union.npz"
        assert not mask_file.exists(), "Preserve previously generated assets"
        np.savez(mask_file, **masks)
        audit["roi"]["recovered_union_file"] = str(mask_file)
        audit["roi"]["recovered_union_file_sha256"] = sha(mask_file)
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(425), transforms.CenterCrop(425),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        inputs = []
        for file in a.images:
            with Image.open(file) as im:
                mode = im.mode
                x = transform(im.convert("RGB"))
            padded = torch.zeros((3, 434, 434), dtype=torch.float32)
            padded[:, :425, :425] = x
            inputs.append(padded)
            audit["images"].append({"path": str(file), "mode": mode, "sha256": sha(file)})
        batch = torch.stack(inputs).to(a.device)
        # Redirect only the original constructor's fixed filesystem path. No
        # original source edits, downloads, shared cache moves, or model rewrite.
        original_hub_load = torch.hub.load
        backbone_state = torch.load(a.dino_weight, map_location="cpu", weights_only=True)
        def local_dino(repo, model, *args, **kwargs):
            assert model == "dinov2_vitb14" and kwargs == {"source": "local"}
            assert repo == "/home/eh2976/.cache/torch/hub/facebookresearch_dinov2_main"
            backbone = original_hub_load(str(a.dino_source), model, source="local", pretrained=False)
            backbone.load_state_dict(backbone_state, strict=True)
            return backbone
        torch.hub.load = local_dino
        sys.path.insert(0, str(a.encoder_source))
        from models.brain_encoder import brain_encoder
        inventory = json.loads((a.output.parent / "encoder_manifest.json").read_text())
        ensemble_parts = {h: {"batch": [], "single": [], "corr": []} for h in ["lh", "rh"]}
        for layer in [1, 3, 5, 7]:
            for run in [1, 2]:
                for hemi in ["lh", "rh"]:
                    t0 = time.monotonic()
                    weight_file = a.weights / f"enc_{layer}/run_{run}/{hemi}/checkpoint_nonavg.pth"
                    expected = next(v for v in inventory["members"] if v["layer"] == layer and v["run"] == run and v["hemisphere"] == hemi)
                    assert sha(weight_file) == expected["sha256"]
                    checkpoint = torch.load(weight_file, map_location="cpu", weights_only=False)
                    cfg = checkpoint["args"]
                    assert cfg.enc_output_layer == layer and cfg.run == run and cfg.hemi == hemi
                    cfg.device = a.device
                    ps = torch.load(a.encoder_source / f"parcels/schaefer/{hemi}_labels_s01.pt", map_location="cpu", weights_only=True)
                    cover = torch.zeros(163842, dtype=torch.int64)
                    for ids in ps:
                        cover[ids] += 1
                    assert len(ps) == 501 and torch.all(cover == 1)
                    dataset = SimpleNamespace(num_parcels=len(ps), parcels=ps,
                        num_hemi_voxels=163842, valid_voxel_mask=cover.bool())
                    model = brain_encoder(cfg, dataset)
                    full_state = {"backbone_model.0.backbone." + k: v for k, v in backbone_state.items()}
                    assert not (set(full_state) & set(checkpoint["model"]))
                    full_state.update(checkpoint["model"])
                    missing = sorted(set(model.state_dict()) - set(full_state))
                    unexpected = sorted(set(full_state) - set(model.state_dict()))
                    assert not missing and not unexpected, (missing, unexpected)
                    model.load_state_dict(full_state, strict=True)
                    assert all(torch.equal(model.state_dict()[k].cpu(), v) for k, v in full_state.items())
                    model.to(a.device).eval()
                    with torch.inference_mode():
                        batched = model(batch)["pred"].cpu()
                        single = torch.cat([model(batch[i:i+1])["pred"].cpu() for i in range(2)])
                    assert batched.shape == (2, 163842) and torch.isfinite(batched).all() and torch.isfinite(single).all()
                    error = float((batched - single).abs().max())
                    assert torch.allclose(batched, single, rtol=2e-4, atol=2e-4), error
                    corr = np.load(weight_file.parent / f"{hemi}_val_corr_nonavg.npy")
                    assert corr.shape == (163842,) and np.isfinite(corr).all()
                    part = ensemble_parts[hemi]
                    part["batch"].append(batched.numpy()); part["single"].append(single.numpy()); part["corr"].append(corr)
                    row = {"layer": layer, "run": run, "hemisphere": hemi,
                        "checkpoint_sha256": expected["sha256"], "full_state_keys": len(full_state),
                        "checkpoint_keys": len(checkpoint["model"]), "backbone_keys": len(backbone_state),
                        "strict_load": True, "all_loaded_tensors_exact": True,
                        "missing_keys": missing, "unexpected_keys": unexpected,
                        "finite_full_vertex_outputs": True, "batch_vs_single_max_abs": error,
                        "elapsed_seconds": time.monotonic() - t0}
                    audit["members"].append(row); save_partial()
                    print(json.dumps(row), flush=True)
                    del model, checkpoint, full_state, batched, single
                    gc.collect(); torch.cuda.empty_cache()
        ensemble = {}
        for h, part in ensemble_parts.items():
            weights = torch.softmax(20 * torch.from_numpy(np.stack(part["corr"])), dim=0).numpy()
            assert np.allclose(weights.sum(axis=0), 1, atol=1e-6)
            b = (np.stack(part["batch"]) * weights[:, None, :]).sum(axis=0)
            s = (np.stack(part["single"]) * weights[:, None, :]).sum(axis=0)
            assert np.allclose(b, s, rtol=2e-4, atol=2e-4)
            ensemble[h] = {"members": 8, "weight_formula": "softmax(20 * val_corr, axis=model)",
                "weights_finite": bool(np.isfinite(weights).all()), "batch_vs_single_max_abs": float(np.abs(b-s).max())}
        audit["ensemble"] = ensemble
        audit["status"] = "public_encoder_forward_passed_formal_retrieval_not_started"
        audit["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        a.output.write_text(json.dumps(audit, indent=2) + "\n")
        print(audit["status"], flush=True)
    except Exception as error:
        audit["status"] = "diagnostic_failed"
        audit["error"] = repr(error)
        save_partial()
        raise


if __name__ == "__main__":
    main()
