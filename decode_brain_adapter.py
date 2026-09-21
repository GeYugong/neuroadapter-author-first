"""
Brain Adapter Decoding Script

This script uses a trained brain adapter model to decode/generate images from fMRI brain signals.
It loads the trained model weights and performs image generation using brain-derived conditioning tokens.

The decoding pipeline:
1. Loads trained brain adapter and guidance generator models
2. Processes fMRI brain data to generate conditioning tokens
3. Uses diffusion model with brain conditioning to generate multiple candidate images
4. Evaluates candidates using brain encoder correlation and selects best match
"""

# Standard library imports
import os
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace
import warnings
import time
import json
import hashlib

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torchvision import transforms

# Diffusion imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# Project-specific imports
from brain_adapter.ip_adapter.ip_adapter import ImageProjModel
from brain_adapter.ip_adapter.utils import is_torch2_available

# Conditional imports based on torch version
if is_torch2_available():
    from brain_adapter.ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor, 
        AttnProcessor2_0 as AttnProcessor
    )
else:
    from brain_adapter.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from brain_adapter.model import GuidanceGenerator
from brain_adapter.dataset import nsd_topk_parcel_dataset  # multi-subject removed

# os.chdir("/engram/nklab/pf2477/brain_decoding/whole_brain_encoder")
# Brain encoder is imported only when candidate selection is requested.

# os.chdir("/engram/nklab/pf2477/brain_decoding/")

warnings.filterwarnings("ignore")

class BrainIPAdapter(torch.nn.Module):
    """
    Brain-adapted IP-Adapter for fMRI-guided image generation.
    
    This class wraps the UNet with IP-Adapter components and provides methods
    for loading trained weights and performing classifier-free guidance.
    """

    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.ip_ckpt = ckpt_path

        if ckpt_path is not None:
            self.load_ip_adapter()
            
        self.cross_attn_maps = {}  # Store cross-attention outputs
        self._register_block_hook()
        
    def _register_block_hook(self):
        pass

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, brain_embeds):
        """
        Forward pass with classifier-free guidance.
        
        Args:
            noisy_latents: Noisy image latents
            timesteps: Diffusion timesteps
            encoder_hidden_states: Text embeddings
            brain_embeds: Brain-derived conditioning tokens
            
        Returns:
            Tuple of (unconditional_pred, conditional_pred)
        """
        # Create conditional and unconditional brain embeddings
        brain_embeds_cond = brain_embeds
        brain_embeds_uncond = torch.zeros_like(brain_embeds)
        
        # Project brain embeddings to IP tokens
        ip_tokens_cond = self.image_proj_model(brain_embeds_cond)
        ip_tokens_uncond = self.image_proj_model(brain_embeds_uncond)
        
        # Concatenate with text embeddings
        encoder_hidden_states_cond = torch.cat([encoder_hidden_states, ip_tokens_cond], dim=1)
        encoder_hidden_states_uncond = torch.cat([encoder_hidden_states, ip_tokens_uncond], dim=1)
        
        # Batch conditional and unconditional for efficient inference
        combined_encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_cond], dim=0)
        combined_noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0)
        
        # Handle timestep broadcasting for the combined batch
        if isinstance(timesteps, torch.Tensor):
            combined_timesteps = torch.cat([timesteps, timesteps], dim=0) if timesteps.dim() > 0 else timesteps
        else:
            combined_timesteps = timesteps
        
        # Single forward pass through U-Net
        combined_noise_pred = self.unet(combined_noisy_latents, combined_timesteps, combined_encoder_hidden_states).sample
        
        # Split predictions
        noise_pred_uncond, noise_pred_cond = combined_noise_pred.chunk(2)
        return noise_pred_uncond, noise_pred_cond

    def load_from_checkpoint(self, ckpt_path: str):
        """Load model weights from checkpoint with verification."""
        # Calculate original checksums for verification
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Load state dictionary
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load weights
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Verify weights changed
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        assert orig_ip_proj_sum != new_ip_proj_sum, "Image projection model weights did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Adapter module weights did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    def load_ip_adapter(self):
        """Load IP-Adapter weights from checkpoint file."""
        try:
            if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
                from safetensors import safe_open
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(self.ip_ckpt, map_location="cpu")
                
            self.image_proj_model.load_state_dict(state_dict["image_proj"])
            ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])
            print(f"Successfully loaded IP-Adapter weights from {self.ip_ckpt}")
        except Exception as e:
            print(f"Warning: Could not load IP-Adapter weights: {e}")

    def set_scale(self, scale):
        """
        Set IP-Adapter attention scales per transformer block.

        Args:
            scale: Can be:
                - float: Global scale for all blocks
                - dict: Granular control per block type and layer
        """
        if not isinstance(scale, (float, dict)):
            raise ValueError("Scale must be float or dict")
            
        for name, attn_processor in self.unet.attn_processors.items():
            if isinstance(attn_processor, IPAttnProcessor):
                if isinstance(scale, float):
                    attn_processor.scale = scale
                else:
                    # Handle dictionary-based scaling
                    for block_type, blocks in scale.items():
                        if name.startswith(block_type):
                            for block_name, block_scale in blocks.items():
                                if block_name in name:
                                    attn_processor.scale = block_scale


def load_diffusion_models(model_path="runwayml/stable-diffusion-v1-5"):
    """Load pre-trained Stable Diffusion components."""
    print("Loading Stable Diffusion components...")
    
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    
    return noise_scheduler, tokenizer, text_encoder, vae, unet


def setup_ip_adapter_modules(args, unet, num_tokens=200):
    """Initialize IP-Adapter attention processor modules."""
    print("Setting up IP-Adapter modules...")
    
    # Create image projection model
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=args.condition_dim,
        clip_extra_context_tokens=num_tokens,
    )

    # Initialize attention processors
    attn_procs = {}
    unet_sd = unet.state_dict()
    
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        # Determine hidden size based on block type
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=num_tokens,
            )
            attn_procs[name].load_state_dict(weights)
    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    return image_proj_model, adapter_modules

def freeze_models(*models):
    """Freeze all parameters in the given models for inference."""
    for model in models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

def create_test_dataset(
    args,
    tokenizer, 
    topk=100, 
    eval_full_dataset=False,
    start_idx=0, 
    end_idx=8, 
    max_samples=None,
    batch_size=1, 
):
    """
    Create test dataset and dataloader with flexible evaluation options.
    
    Args:
        tokenizer: CLIP tokenizer for text processing
        topk: Number of top parcels to use per hemisphere
        eval_full_dataset: If True, use entire test dataset
        start_idx: Starting index (only used if eval_full_dataset=False)
        end_idx: Ending index (only used if eval_full_dataset=False)
        max_samples: Maximum number of samples to evaluate (None = no limit)
        batch_size: Batch size for dataloader
        
    Returns:
        Tuple of (test_dataset, test_dataloader, actual_indices)
    """
    dataset_args = SimpleNamespace(
        subj=args.subject_id,
        backbone_arch="dinov2_q",
        data_dir=args.data_dir,
        imgs_dir=args.imgs_dir,
        parcel_dir=args.parcel_dir,
        hemi=None,
        tokenizer=tokenizer,
        gen_size=512,
    )

    test_dataset = nsd_topk_parcel_dataset(dataset_args, split=args.split, transform=None, topk=topk)
    total_samples = len(test_dataset)
    
    print(f"Test dataset created with {total_samples} total samples, using top {topk} parcels per hemisphere.")
    
    # Determine evaluation indices
    if eval_full_dataset:
        print("Evaluating on FULL test dataset")
        indices = list(range(total_samples))
        if max_samples is not None and max_samples < total_samples:
            indices = indices[:max_samples]
            print(f"Limited to first {max_samples} samples")
        actual_start, actual_end = 0, len(indices)
    else:
        print(f"Evaluating on SUBSET: indices {start_idx} to {end_idx}")
        # Validate indices
        if end_idx > total_samples:
            print(f"Warning: end_idx ({end_idx}) exceeds dataset size ({total_samples}). Adjusting to {total_samples}")
            end_idx = total_samples
        if start_idx >= total_samples:
            raise ValueError(f"start_idx ({start_idx}) exceeds dataset size ({total_samples})")
        if start_idx >= end_idx:
            raise ValueError(f"start_idx ({start_idx}) must be less than end_idx ({end_idx})")
            
        indices = list(range(start_idx, end_idx))
        if max_samples is not None and len(indices) > max_samples:
            indices = indices[:max_samples]
            print(f"Limited to first {max_samples} samples from specified range")
        actual_start, actual_end = indices[0], indices[-1] + 1

    print(f"Final evaluation: {len(indices)} samples (indices {actual_start} to {actual_end-1})")

    if args.image_ids:
        requested = np.loadtxt(args.image_ids, dtype=np.int64).reshape(-1)
        mapping = {int(v):i for i,v in enumerate(test_dataset.base_dataset.split_presented_imgs)}
        indices = [mapping[int(v)] for v in requested]
        actual_start, actual_end = 0, len(indices)
    # Create subset and dataloader
    test_subset = torch.utils.data.Subset(test_dataset, indices)
    test_dataloader = torch.utils.data.DataLoader(
        test_subset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=(torch.cuda.is_available()),
    )
    
    return test_dataset, test_dataloader, (actual_start, actual_end)


def setup_device():
    """Initialize device for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def validate_arguments(args):
    """Validate and process command line arguments."""
    # Handle evaluation mode logic
    if args.eval_full_dataset:
        print("Mode: Full dataset evaluation")
        if args.max_samples:
            print(f"  - Limited to first {args.max_samples} samples")
    else:
        print(f"Mode: Index-based evaluation ({args.start_idx} to {args.end_idx})")
        
    # Validate numerical arguments
    if args.num_predictions <= 0:
        raise ValueError("num_predictions must be positive")
    if args.noise_factor < 0:
        raise ValueError("noise_factor must be non-negative")
    if args.topk <= 0:
        raise ValueError("topk must be positive")
        
    return args


def estimate_memory_usage(num_samples, num_predictions, image_size=512):
    """Estimate memory usage for the evaluation."""
    # Rough estimates in GB
    base_model_memory = 8.0  # Base models (UNet, VAE, etc.)
    brain_encoder_memory = 4.0  # Brain encoder
    
    # Per sample memory (images + features)
    bytes_per_pixel = 4  # float32
    image_memory_per_sample = (num_predictions * image_size * image_size * 3 * bytes_per_pixel) / (1024**3)
    
    total_memory = base_model_memory + brain_encoder_memory + (num_samples * image_memory_per_sample)
    
    print(f"Estimated memory usage: {total_memory:.1f} GB")
    if total_memory > 16:
        print("Warning: High memory usage expected. Consider reducing batch_size or num_predictions")
    
    return total_memory


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def save_individual_results(
    original_images, 
    decoded_images, 
    candidate_images_list,
    correlation_scores_list,
    save_dir, 
    args, 
    evaluation_indices,
    dataset_indices,
    processing_time=None,
    state=None,
    finalize=True
):
    """
    Save results individually for each sample with organized folder structure.
    
    Supports streaming usage by reusing the returned state between calls and
    writing metadata/summary only when finalize=True.
    """
    if state is None and original_images is None and finalize:
        return None

    if state is None:
        epoch_dir = os.path.join(save_dir, f"epoch_{args.saved_epochs}")
        Path(epoch_dir).mkdir(parents=True, exist_ok=True)

        if args.eval_full_dataset:
            eval_dir = os.path.join(epoch_dir, "full")
            mode_name = "full_dataset"
        else:
            eval_dir = os.path.join(epoch_dir, "subset")
            mode_name = f"subset_{evaluation_indices[0]}_{evaluation_indices[1]-1}"

        Path(eval_dir).mkdir(parents=True, exist_ok=True)

        global_metadata = {
            'model_weights_dir': args.model_weights_dir,
            'saved_epochs': args.saved_epochs,
            'num_predictions': args.num_predictions,
            'noise_factor': args.noise_factor,
            'topk': args.topk,
            'condition_dim': args.condition_dim,
            'num_decoder_queries': args.num_decoder_queries,
            'sub_approach': args.sub_approach,
            'subject_id': args.subject_id,
            'evaluation_mode': 'full_dataset' if args.eval_full_dataset else 'indices',
            'start_idx': evaluation_indices[0],
            'end_idx': evaluation_indices[1],
            'num_samples': 0,
            'timestamp': np.datetime64('now').astype(str),
            'precision': args.precision,
            'denoising_steps': args.denoising_steps,
        }

        state = {
            'eval_dir': eval_dir,
            'metadata_path': os.path.join(eval_dir, "evaluation_metadata.json"),
            'summary_path': os.path.join(eval_dir, "sample_summary.json"),
            'dataset_indices': [],
            'sample_filenames': [],
            'num_samples': 0,
            'global_metadata': global_metadata,
            'mode_name': mode_name,
            'evaluation_indices': evaluation_indices,
        }

    sample_count = len(original_images) if original_images is not None else 0
    if sample_count:
        candidate_images_list = candidate_images_list or [None] * sample_count
        correlation_scores_list = correlation_scores_list or [None] * sample_count
        assert len(dataset_indices) == sample_count, "Dataset indices must match sample count"

        for idx in range(sample_count):
            orig_img = original_images[idx]
            decoded_img = decoded_images[idx]
            dataset_idx = int(dataset_indices[idx])
            corr_scores = correlation_scores_list[idx]
            corr_score_value = float(np.max(corr_scores)) if corr_scores is not None else None

            sample_filename = f"sample_{dataset_idx:06d}.npz"
            sample_path = os.path.join(state['eval_dir'], sample_filename)
            eval_index = state['num_samples']

            save_data = {
                'groundtruth_image': orig_img.detach().cpu().numpy(),
                'predicted_image': decoded_img.detach().cpu().numpy(),
                'evaluation_index': eval_index,
                'correlation_score': corr_score_value,
            }

            if args.save_all_candidates:
                candidate_imgs = candidate_images_list[idx]
                save_data['candidate_images'] = candidate_imgs
                save_data['correlation_scores'] = corr_scores
                save_data['best_candidate_idx'] = int(np.argmax(corr_scores))

            np.savez_compressed(sample_path, **save_data)

            state['dataset_indices'].append(dataset_idx)
            state['sample_filenames'].append(sample_filename)
            state['num_samples'] += 1

            if state['num_samples'] % 50 == 0:
                print(f"Saved {state['num_samples']} samples so far")

    if finalize and state is not None:
        import json

        metadata = state['global_metadata']
        metadata['num_samples'] = state['num_samples']
        if processing_time is not None and state['num_samples'] > 0:
            metadata['processing_time_seconds'] = processing_time
            metadata['time_per_sample'] = processing_time / max(state['num_samples'], 1)

        with open(state['metadata_path'], 'w') as f:
            json.dump(metadata, f, indent=2)

        summary_data = {
            'evaluation_indices': state['evaluation_indices'],
            'dataset_indices': state['dataset_indices'],
            'sample_filenames': state['sample_filenames'],
            'num_samples': state['num_samples'],
            'mode': state['mode_name']
        }
        with open(state['summary_path'], 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nResults saved successfully!")
        print(f"Directory: {state['eval_dir']}")
        print(f"Samples: {state['num_samples']}")
        print(f"Metadata: {state['metadata_path']}")
        print(f"Summary: {state['summary_path']}")

        return state['eval_dir']

    return state


def run_diffusion(brain_embeds, img_ip, models_dict, num_predictions=1, noise_factor=4.0, denoising_steps=50, generator=None):
    """
    Run diffusion process with brain conditioning to generate images.
    
    Args:
        brain_embeds: Brain-derived conditioning embeddings
        img_ip: Initial image for latent space (set to zeros for unconditional)
        models_dict: Dictionary containing all necessary models
        num_predictions: Number of candidate images to generate
        noise_factor: Classifier-free guidance scale
        
    Returns:
        Generated images as numpy array
    """
    # Extract models from dictionary
    tokenizer = models_dict['tokenizer']
    text_encoder = models_dict['text_encoder']
    vae = models_dict['vae']
    noise_scheduler = models_dict['noise_scheduler']
    brain_adapter = models_dict['brain_adapter']
    device = models_dict['device']
    weight_dtype = models_dict['weight_dtype']
    
    # Process empty text prompt
    text = ""
    text_input_ids = tokenizer(
        text,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    
    encoder_hidden_states = text_encoder(text_input_ids)[0].to(device, dtype=weight_dtype)
    encoder_hidden_states = encoder_hidden_states.repeat(num_predictions, 1, 1)
    
    # Clean up text input ids immediately
    del text_input_ids
    torch.cuda.empty_cache()

    # Setup diffusion timesteps
    num_inference_steps = denoising_steps
    noise_scheduler.set_timesteps(num_inference_steps)
    init_timestep = min(int(num_inference_steps * 1.0), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = noise_scheduler.timesteps[t_start:]

    # Generate initial latents from input image
    init_latents = vae.encode(img_ip).latent_dist.sample(generator=generator) * vae.config.scaling_factor

    # Generate different noise for each prediction
    latents = []
    for i in range(num_predictions):
        noise = torch.randn(init_latents.shape, generator=generator).to(device, dtype=weight_dtype)
        noised_latents = noise_scheduler.add_noise(init_latents, noise, timesteps[:1])
        latents.append(noised_latents)
        # Clean up noise tensor immediately
        del noise
    latents = torch.cat(latents, dim=0)

    # Diffusion denoising process
    for i, t in enumerate(tqdm(timesteps, desc="Diffusion steps", disable=True)):
        noise_pred_uncond, noise_pred_cond = brain_adapter(
            latents, t, encoder_hidden_states, brain_embeds.repeat(num_predictions, 1, 1)
        )
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + noise_factor * (noise_pred_cond - noise_pred_uncond)
        
        # Clean up intermediate predictions
        del noise_pred_uncond, noise_pred_cond
        
        latents = noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
        
        # Clean up noise prediction
        del noise_pred
        
        # More frequent memory cleanup during diffusion
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Decode latents to images
    latents = 1 / vae.config.scaling_factor * latents
    with torch.autocast(device_type=device.type):
        pred_images = vae.decode(latents).sample

    # Post-process images and move to CPU immediately
    pred_images = (pred_images / 2 + 0.5).clamp(0, 1)
    pred_images = pred_images.cpu().permute(0, 2, 3, 1).numpy()
    pred_images = (pred_images * 255).round().astype("uint8")
    
    # Clean up all intermediate tensors
    del latents, init_latents, encoder_hidden_states
    torch.cuda.empty_cache()

    return pred_images



def compute_brain_correlation(ground_truth, predictions):
    """
    Compute Pearson correlation between ground truth and predicted brain activity.
    
    Args:
        ground_truth: Ground truth brain activity [1, voxels]
        predictions: Predicted brain activity [N, voxels]
        
    Returns:
        Correlation scores for each prediction [N]
    """
    # Standardize both ground truth and predictions with epsilon for numerical stability
    eps = 1e-8
    gt_std = ground_truth.std(dim=1, keepdim=True) + eps
    pred_std = predictions.std(dim=1, keepdim=True) + eps
    
    gt_centered = (ground_truth - ground_truth.mean(dim=1, keepdim=True)) / gt_std
    pred_centered = (predictions - predictions.mean(dim=1, keepdim=True)) / pred_std

    # Compute Pearson correlation
    correlations = torch.matmul(pred_centered, gt_centered.T).squeeze() / ground_truth.size(1)
    return correlations


def preprocess_image(image, target_size, device="cpu"):
    """Preprocess image to target size."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])

    img_transformed = transform(image)
    return img_transformed.to(device)


def process_brain_data_for_correlation(batch, test_dataset, brain_encoder_predictions, device):
    """
    Process brain data and predictions to compute correlation scores.
    
    Args:
        batch: Data batch containing brain activity
        test_dataset: Test dataset with parcel information
        brain_encoder_predictions: Predicted brain activity from encoder
        device: Device to use for computation
        
    Returns:
        Tuple of (ground_truth_concat, predictions_concat) ready for correlation
    """
    target_device = device
    
    # Concatenate brain data across all parcels and hemispheres
    ground_truth_concat = []
    predictions_concat = []
    
    for hemisphere in ["lh", "rh"]:
        for parcel_idx, parcel_id in enumerate(test_dataset.selected_parcel_idx[hemisphere]):

            voxel_indices = test_dataset.parcels[hemisphere][parcel_id]
            num_voxels = len(voxel_indices)
            
            # Ground truth brain activity for this parcel
            ground_truth_concat.append(
                batch[f"brain_{hemisphere}_f"][:, parcel_idx, :num_voxels].to(target_device)
            )
            
            # Predicted brain activity for this parcel
            predictions_concat.append(
                brain_encoder_predictions[hemisphere][:, voxel_indices].to(target_device)
            )
    
    # Concatenate all parcels
    ground_truth = torch.cat(ground_truth_concat, dim=1)
    predictions = torch.cat(predictions_concat, dim=1)
    
    assert ground_truth.shape[1] == predictions.shape[1], \
        f"Dimension mismatch: GT {ground_truth.shape[1]} vs Pred {predictions.shape[1]}"
    
    return ground_truth, predictions


def decode_images_from_brain(
    dataloader, models_dict, brain_encoder, test_dataset, num_predictions=8,
    noise_factor=4.0, save_all_candidates=False, denoising_steps=50, save_config=None
):
    """
    Main function to decode images from brain signals with streaming saves.

    Args:
        dataloader: DataLoader containing brain data
        models_dict: Dictionary with all necessary models
        brain_encoder: Brain encoder for validation
        test_dataset: Dataset instance for parcel information
        num_predictions: Number of candidate images per brain sample
        noise_factor: Guidance scale for diffusion
        save_all_candidates: Whether to save all candidate images (for subset evaluation)
        denoising_steps: Number of diffusion denoising steps
        save_config: Dict with keys: base_save_dir, args, evaluation_indices, start_time
        
    Returns:
        Dict with keys: results_dir, num_samples, processing_time
    """
    results_state = None
    samples_processed = 0
    if 'start_time' not in save_config:
        save_config['start_time'] = time.time()

    device = models_dict['device']
    weight_dtype = models_dict['weight_dtype']
    guidance_generator = models_dict['guidance_generator']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Decoding brain signals")):
            img = batch["img_encoder"].squeeze()
            img_cpu = img.cpu()

            if hasattr(dataloader.dataset, 'indices'):
                dataset_idx = dataloader.dataset.indices[batch_idx]
            else:
                dataset_idx = batch_idx
            dataset_idx = int(dataset_idx)
            
            # Process brain data
            lh = batch["brain_lh_f"].to(device, dtype=weight_dtype)
            rh = batch["brain_rh_f"].to(device, dtype=weight_dtype)
            brain_data = torch.cat([lh, rh], dim=1)
            
            del lh, rh
            
            # Validate brain data dimensions
            assert brain_data.shape[1] == test_dataset.num_parcels, \
                f"Expected {test_dataset.num_parcels} parcels, got {brain_data.shape[1]}"
            assert brain_data.shape[2] == test_dataset.max_voxels, \
                f"Expected {test_dataset.max_voxels} voxels, got {brain_data.shape[2]}"

            # Generate brain conditioning tokens
            brain_embeds, _ = guidance_generator(brain_data)
            del brain_data

            # Use zero-filled image as initialization (no image information)
            img_init = batch["img_ipadapter"].to(device, dtype=weight_dtype)
            img_init = torch.zeros_like(img_init)

            # Generate candidate images using diffusion
            candidate_images = run_diffusion(
                brain_embeds, img_init, models_dict,
                num_predictions=num_predictions, noise_factor=noise_factor,
                denoising_steps=denoising_steps
            )
            
            del brain_embeds, img_init
            torch.cuda.empty_cache()

            # Get brain encoder predictions for all candidates
            brain_predictions = brain_encoder.forward(candidate_images)

            # Compute correlations with ground truth brain activity
            ground_truth, predictions = process_brain_data_for_correlation(
                batch, test_dataset, brain_predictions, device
            )
            correlations = compute_brain_correlation(ground_truth, predictions)
            correlations_numpy = correlations.cpu().numpy()
            best_idx = int(np.argmax(correlations_numpy))
            best_image = candidate_images[best_idx]
            best_image = preprocess_image(best_image, img.shape[-1], "cpu")

            # Save results incrementally in the batch loop
            candidates_payload = [candidate_images] if save_all_candidates else [None]
            correlations_payload = [correlations_numpy]
            results_state = save_individual_results(
                [img_cpu], [best_image], candidates_payload, correlations_payload,
                save_config['base_save_dir'], save_config['args'], save_config['evaluation_indices'],
                [dataset_idx], state=results_state, finalize=False
            )
            samples_processed += 1
            
            # Clean up all GPU tensors for this iteration
            del brain_predictions, ground_truth, predictions, correlations
            del candidate_images
            
            # Force garbage collection every few iterations
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Print memory usage periodically for monitoring
            if (batch_idx + 1) % 20 == 0 and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"  Batch {batch_idx + 1}: GPU memory allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB")

    # Finalize streaming save with metadata
    processing_time = time.time() - save_config['start_time']
    results_dir = None
    if results_state is not None:
        results_dir = save_individual_results(
            None, None, None, None,
            save_config['base_save_dir'], save_config['args'], save_config['evaluation_indices'],
            [], processing_time=processing_time, state=results_state, finalize=True
        )
    
    return {
        'results_dir': results_dir,
        'num_samples': samples_processed,
        'processing_time': processing_time,
    }


def main():
    """Main execution function with enhanced evaluation options."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Setup device
    device = setup_device()
    
    start_time = time.time()
    
    # Parse and validate arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    args = validate_arguments(args)
    
    print("=== Brain Adapter Decoding Configuration ===")
    print(f"Model weights: {args.model_weights_dir}")
    print(f"Checkpoint: {args.saved_epochs}")
    print(f"Evaluation mode: {'Full dataset' if args.eval_full_dataset else f'Indices {args.start_idx}-{args.end_idx}'}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'No limit'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Noise factor: {args.noise_factor}")
    print(f"Predictions per sample: {args.num_predictions}")
    print(f"Denoising steps: {args.denoising_steps}")
    print(f"Top-k parcels: {args.topk}")
    print("=" * 50)

    # For full dataset evaluation, reduce batch size and predictions to save memory
    if args.eval_full_dataset and args.batch_size > 1:
        print(f"Warning: For full dataset evaluation, consider using batch_size=1 to avoid memory issues")
    
    if args.eval_full_dataset and args.num_predictions > 4:
        print(f"Warning: For full dataset evaluation, consider reducing num_predictions to 4 or less")
    
    # Load pre-trained models
    print("Loading diffusion models...")
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_diffusion_models(args.pretrained_model_name_or_path)
    
    # Setup IP-Adapter components
    print("Setting up IP-Adapter...")
    image_proj_model, adapter_modules = setup_ip_adapter_modules(args, unet, num_tokens=args.topk*2)
    
    # Create test dataset with flexible evaluation
    print("Creating test dataset...")
    test_dataset, test_dataloader, evaluation_indices = create_test_dataset(
        args,
        tokenizer, 
        topk=args.topk, 
        eval_full_dataset=args.eval_full_dataset,
        start_idx=args.start_idx, 
        end_idx=args.end_idx,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Estimate memory usage
    num_samples = evaluation_indices[1] - evaluation_indices[0]
    estimate_memory_usage(num_samples, args.num_predictions)
    
    # Create brain adapter and guidance generator
    print("Initializing brain models...")
    brain_adapter = BrainIPAdapter(unet, image_proj_model, adapter_modules, ckpt_path=None)
    guidance_generator = GuidanceGenerator(
        num_parcels=test_dataset.num_parcels,
        max_voxels=test_dataset.max_voxels,
        num_decoder_queries=args.num_decoder_queries,
        output_dim=args.condition_dim,
        sub_approach=args.sub_approach,
    )

    if args.export_path:
        saved_vertices = torch.load(Path(args.export_path).parent/'parcel_vertices.pt',weights_only=True)
        for h in ['lh','rh']:
            assert all(torch.equal(x,y) for x,y in zip(saved_vertices[h],test_dataset.get_selected_voxel_indices()[h]))
    # Load trained model weights
    print(f"Loading trained weights from checkpoint-{args.saved_epochs}...")
    if args.export_path:
        weights = torch.load(args.export_path,map_location='cpu',weights_only=True)
        brain_adapter.image_proj_model.load_state_dict(weights['image_proj'],strict=True)
        brain_adapter.adapter_modules.load_state_dict(weights['ip_adapter'],strict=True)
        guidance_generator.load_state_dict(weights['guidance_generator'],strict=True)
        del weights
    else:
        brain_adapter.load_state_dict(torch.load(f"{args.model_weights_dir}/checkpoint-{args.saved_epochs}/pytorch_model.bin",map_location='cpu',weights_only=True))
        guidance_generator.load_state_dict(torch.load(f"{args.model_weights_dir}/checkpoint-{args.saved_epochs}/pytorch_model_1.bin",map_location='cpu',weights_only=True))

    # Set up weight dtype and move models to device
    weight_dtype = torch.float32
    
    # Move models to device
    print("Moving models to device...")
    brain_adapter = brain_adapter.to(device)
    guidance_generator = guidance_generator.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    image_proj_model = image_proj_model.to(device)

    print(f"Models moved to device: {device}")
    print(f"Target weight dtype: {weight_dtype}")
    print("Models prepared successfully")

    # Clear any cached memory after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3
        print(f"Initial GPU memory usage: {initial_memory:.2f}GB")

    brain_encoder = None
    if not args.no_selection:
        from whole_brain_encoder.brain_encoder_wrapper import BrainEncoderWrapper
        brain_encoder = BrainEncoderWrapper(num_gpus=args.num_gpus,subj=args.subject_id)

    # Freeze all models for inference
    print("Freezing models for inference...")
    freeze_models(unet, vae, text_encoder, image_proj_model, adapter_modules, brain_adapter, guidance_generator)

    # Create models dictionary for easy passing
    models_dict = {
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'vae': vae,
        'noise_scheduler': noise_scheduler,
        'brain_adapter': brain_adapter,
        'guidance_generator': guidance_generator,
        'device': device,
        'weight_dtype': weight_dtype,
    }

    if args.no_selection:
        assert args.batch_size == 1 and args.num_predictions == 2
        destination = Path(args.decoded_stimuli_dir)
        destination.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():
            for batch in tqdm(test_dataloader,desc=f'Decode {args.split}'):
                image_id = int(batch['image_id'].item())
                target = destination/f'image_{image_id:06d}.npz'
                if target.exists():
                    raise FileExistsError(f'Refusing to replace decoded result: {target}')
                brain_data = torch.cat([batch['brain_lh_f'],batch['brain_rh_f']],dim=1).to(device,dtype=weight_dtype)
                brain_embeds,_ = guidance_generator(brain_data)
                # Shape is fixed by protocol; no GT values enter the generator.
                img_init = torch.zeros((1,3,512,512),device=device,dtype=weight_dtype)
                candidates=[]
                seeds=[]
                for candidate in range(2):
                    seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{image_id}:{candidate}'.encode()).digest()[:8],'little') % (2**63-1)
                    generator = torch.Generator(device='cpu').manual_seed(seed)
                    images = run_diffusion(brain_embeds,img_init,models_dict,num_predictions=1,
                                           noise_factor=args.noise_factor,denoising_steps=args.denoising_steps,generator=generator)
                    candidates.append(images[0]); seeds.append(seed)
                np.savez_compressed(target,image_id=image_id,groundtruth_image=batch['img_encoder'][0].cpu().numpy(),
                                    candidate_images=np.stack(candidates),seeds=np.asarray(seeds,dtype=np.int64))
        (destination/'protocol.json').write_text(json.dumps(dict(split=args.split,image_ids=args.image_ids,
            candidate_batch=1,num_predictions=2,denoising_steps=args.denoising_steps,cfg=args.noise_factor,
            seed=args.seed,selection=False,gt_initialization=False,weight_dtype=str(weight_dtype),
            vae_decode_autocast='author CUDA default float16',export_path=args.export_path),indent=2))
        return 0
    # Run decoding
    print(f"\nStarting image decoding from brain signals...")
    print(f"Processing {num_samples} samples with {args.num_predictions} predictions each...")
    
    # # Determine whether to save all candidates
    # save_candidates = args.save_all_candidates and not args.eval_full_dataset
    # if args.save_all_candidates and args.eval_full_dataset:
    #     print("Warning: --save_all_candidates ignored for full dataset evaluation due to storage constraints")
    save_candidates = args.save_all_candidates

    brain_decoding_dir = "/engram/nklab/pf2477/brain_decoding/"
    if not os.path.exists(brain_decoding_dir):
        brain_decoding_dir = os.getcwd()
    os.chdir(brain_decoding_dir)
    base_save_dir = os.path.join(args.decoded_stimuli_dir, args.model_weights_dir.split("/")[-1])

    # Setup save configuration for streaming mode
    save_context = {
        'base_save_dir': base_save_dir,
        'args': args,
        'evaluation_indices': evaluation_indices,
        'start_time': start_time,
    }

    # Call decode_images_from_brain with save_config for incremental saving
    results_output = decode_images_from_brain(
        test_dataloader, models_dict, brain_encoder, test_dataset,
        num_predictions=args.num_predictions, noise_factor=args.noise_factor,
        save_all_candidates=save_candidates, denoising_steps=args.denoising_steps,
        save_config=save_context
    )
    print("Decoding completed successfully")
    cleanup_gpu_memory()

    # Extract results
    processing_time = results_output.get('processing_time', time.time() - start_time)
    num_samples = results_output.get('num_samples', 0)
    results_dir = results_output.get('results_dir')

    if num_samples > 0:
        print(f"Total processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
        print(f"Time per sample: {processing_time/num_samples:.1f} seconds")
    else:
        print("No samples were processed.")
    
    if results_dir:
        print(f"\nDecoding completed successfully!")
        print(f"Results directory: {results_dir}")
        print(f"Processed {num_samples} samples in {processing_time/60:.1f} minutes")
        print(f"Individual files saved for each sample")
    else:
        print("No results directory was created.")

    return 0
    
    
def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Decode images from brain signals using trained Brain Adapter model"
    )
    parser.add_argument(
        "--model_weights_dir", 
        type=str, 
        default="brain_adapter/model_weights/06_27_2025-21_15",
        help="Directory containing trained model weights"
    )
    parser.add_argument(
        "--decoded_stimuli_dir", 
        type=str, 
        default="./brain_adapter/decoded_stimuli",
        help="Directory to save decoded images"
    )
    parser.add_argument(
        "--saved_epochs", 
        type=str, 
        default="100",
        help="Training step checkpoint to load"
    )
    parser.add_argument(
        "--num_predictions", 
        type=int, 
        default=8,
        help="Number of candidate images to generate per brain sample"
    )
    
    # Evaluation mode selection
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--eval_full_dataset",
        action="store_true",
        help="Evaluate on the entire test dataset (ignores start_idx/end_idx)"
    )
    eval_group.add_argument(
        "--eval_indices",
        action="store_true", 
        default=True,
        help="Evaluate on specific indices (default behavior)"
    )

    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Starting index for test samples (only used with --eval_indices)"
    )
    parser.add_argument(
        "--end_idx", 
        type=int, 
        default=8,
        help="Ending index for test samples (only used with --eval_indices)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (useful for full dataset evaluation)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = no limit)"
    )
    parser.add_argument(
        "--noise_factor", 
        type=float, 
        default=1.0,
        help="Classifier-free guidance scale for diffusion"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for decoding"
    )
    parser.add_argument(
        "--subject_id",
        type=int,
        default=1,
        help="Subject ID for fMRI data"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Number of top parcels to use per hemisphere"
    )
    parser.add_argument(
        "--num_decoder_queries",
        type=int,
        default=50,
        help="Number of decoder queries for guidance generator"
    )
    parser.add_argument(
        "--condition_dim",
        type=int,
        default=768,
        help="Conditioning dimension for guidance generator"
    )
    parser.add_argument(
        "--sub_approach", 
        type=str, 
        default="linear_projection", 
        help="Sub-approach for training. Options: 'linear_projection', 'masking + transformer_decoder'"
    )
    parser.add_argument(
        "--save_all_candidates",
        action="store_true",
        default=False,
        help="Save all candidate images for subset evaluation (not recommended for full dataset due to storage)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp16", "fp32", "bf16"],
        help="Inference precision (auto tries bf16 then fp16 on CUDA)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers."
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=50,
        help="Number of diffusion denoising steps."
    )

    parser.add_argument('--data_dir',required=True)
    parser.add_argument('--imgs_dir',required=True)
    parser.add_argument('--parcel_dir',required=True)
    parser.add_argument('--pretrained_model_name_or_path',required=True)
    parser.add_argument('--split',choices=['train','val','test'],default='val')
    parser.add_argument('--image_ids')
    parser.add_argument('--no_selection',action='store_true')
    parser.add_argument('--export_path')
    parser.add_argument('--seed',type=int,default=42)
    return parser


if __name__ == "__main__":
    sys.exit(main())
    """
    Usage Examples (multi-subject removed):
    python decode_brain_adapter_2.py \
        --model_weights_dir brain_adapter/model_weights/08_16_2025-18_03 \
        --saved_epochs 200 \
        --start_idx 128 \
        --end_idx 192 \
        --num_predictions 2 \
        --noise_factor 4.0 \
        --precision auto \
        --subject_id 1 \
        --topk 100

    python decode_brain_adapter_2.py \
        --model_weights_dir brain_adapter/model_weights/08_14_2025-00_21 \
        --saved_epochs 200 \
        --eval_full_dataset \
        --num_predictions 4 \
        --noise_factor 2.0 \
        --precision fp16 \
        --subject_id 1 \
        --topk 100
    """