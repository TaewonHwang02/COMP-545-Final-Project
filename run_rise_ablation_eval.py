#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
RISE Ablation Evaluation: Compare performance with/without ViT tokens
Ablation study on a subset of the RISE benchmark
"""

import os
import json
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm

import torch
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

# OpenAI API Key should be set via environment variable for GPT evaluation
# export OPENAI_API_KEY='your-api-key-here'


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rise_samples(metadata_file, num_samples=10):
    """Load samples from RISE dataset."""
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    # RISE data is in list format, not dictionary
    # Random sampling
    sampled_data = random.sample(data, min(num_samples, len(data)))
    
    # Convert to dict format using index as key
    samples = {item['index']: item for item in sampled_data}
    return samples


def load_model(model_path, max_mem_per_gpu="40GiB"):
    """Load BAGEL model."""
    print("=" * 80)
    print("Loading Model")
    print("=" * 80)
    
    # Configuration files
    print("\n[1/8] Loading configuration...")
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # VAE
    print("\n[2/8] Loading VAE...")
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    
    # Bagel configuration
    print("\n[3/8] Creating Bagel configuration...")
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )
    
    # Initialize model
    print("\n[4/8] Initializing model structure...")
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # Tokenizer
    print("\n[5/8] Loading Tokenizer...")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # Image transforms
    print("\n[6/8] Preparing image transforms...")
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    
    # Device mapping
    print("\n[7/8] Setting up device mapping...")
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
    
    # Load weights
    print("\n[8/8] Loading model weights...")
    checkpoint_path = os.path.join(model_path, "ema.safetensors")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()
    
    print("Model loaded successfully!")
    
    # Create inferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )
    
    return inferencer, vae_transform


def generate_image_ablation(inferencer, input_image, prompt, use_vit, inference_hyper):
    """Generate image - ablation version."""
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        # Initialize context
        gen_context = inferencer.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        
        # Add input image
        gen_context = inferencer.update_context_image(
            input_image,
            gen_context,
            vae=True,
            vit=use_vit  # Key parameter for ablation
        )
        
        cfg_text_context = deepcopy(gen_context)
        
        # Add text prompt
        gen_context = inferencer.update_context_text(prompt, gen_context)
        cfg_img_context = inferencer.update_context_text(prompt, cfg_img_context)
        
        # Generate image
        image_shape = input_image.size[::-1]  # (H, W)
        output_image = inferencer.gen_image(
            image_shape,
            gen_context,
            cfg_text_precontext=cfg_text_context,
            cfg_img_precontext=cfg_img_context,
            **inference_hyper
        )
        
        return output_image


def run_ablation_evaluation(args):
    """Run ablation evaluation."""
    print("=" * 80)
    print("RISE Ablation Evaluation: With/Without ViT Comparison")
    print("=" * 80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"rise_ablation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    no_vit_dir = output_dir / "no_vit"
    with_vit_dir = output_dir / "with_vit"
    no_vit_dir.mkdir(exist_ok=True)
    with_vit_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load model
    inferencer, vae_transform = load_model(args.model_path, args.max_mem_per_gpu)
    
    # Inference hyperparameters
    inference_hyper = dict(
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )
    
    # Load RISE samples
    print(f"\n[Data] Loading RISE samples...")
    samples = load_rise_samples(args.metadata_file, args.num_samples)
    print(f"Loaded {len(samples)} samples")
    
    # Run evaluation
    results = []
    
    print("\n" + "=" * 80)
    print("Starting Image Generation")
    print("=" * 80)
    
    for idx, (sample_id, sample_data) in enumerate(tqdm(samples.items(), desc="Processing samples")):
        print(f"\n[Sample {idx+1}/{len(samples)}] ID: {sample_id}")
        
        try:
            # Load input image
            # RISE data format: image field contains relative path, needs 'data' subdirectory
            image_path = os.path.join(args.rise_data_dir, 'data', sample_data.get('image', ''))
            if not os.path.exists(image_path):
                print(f"  Warning: Skipping - image not found {image_path}")
                continue
            
            input_image = Image.open(image_path).convert('RGB')
            input_image = vae_transform.resize_transform(pil_img2rgb(input_image))
            
            # Get prompt
            prompt = sample_data.get('instruction', '')
            print(f"  Prompt: {prompt[:100]}...")
            
            # 1. Generate without ViT
            print(f"  Generating (no ViT)...")
            output_no_vit = generate_image_ablation(
                inferencer, input_image, prompt, use_vit=False, inference_hyper=inference_hyper
            )
            output_no_vit_path = no_vit_dir / f"{sample_id}.png"
            output_no_vit.save(output_no_vit_path)
            
            # 2. Generate with ViT
            print(f"  Generating (with ViT)...")
            output_with_vit = generate_image_ablation(
                inferencer, input_image, prompt, use_vit=True, inference_hyper=inference_hyper
            )
            output_with_vit_path = with_vit_dir / f"{sample_id}.png"
            output_with_vit.save(output_with_vit_path)
            
            # Record results
            results.append({
                'sample_id': sample_id,
                'prompt': prompt,
                'input_image': image_path,
                'output_no_vit': str(output_no_vit_path),
                'output_with_vit': str(output_with_vit_path),
                'category': sample_data.get('category', 'unknown')
            })
            
            print(f"  Done")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Ablation Evaluation Complete!")
    print("=" * 80)
    print(f"""
Results Summary:
  - Successfully processed: {len(results)}/{len(samples)} samples
  - Output directory: {output_dir}
  
Output Files:
  - No ViT results: {no_vit_dir}
  - With ViT results: {with_vit_dir}
  - Results log: {results_file}
  
Next Steps:
  1. Use GPT to evaluate both sets of images
  2. Compare and analyze differences between results
  3. Visualize comparison results
  
GPT Evaluation Commands:
  # Evaluate no-ViT version
  python ./eval/gen/rise/gpt_eval.py \\
    --data {args.metadata_file} \\
    --input {args.rise_data_dir} \\
    --output {no_vit_dir}
  
  # Evaluate with-ViT version  
  python ./eval/gen/rise/gpt_eval.py \\
    --data {args.metadata_file} \\
    --input {args.rise_data_dir} \\
    --output {with_vit_dir}
""")


def main():
    parser = argparse.ArgumentParser(description="RISE Ablation Evaluation")
    
    # Model configuration
    parser.add_argument("--model-path", type=str, default="./checkpoints/BAGEL-7B-MoT",
                        help="Path to model checkpoint")
    parser.add_argument("--max-mem-per-gpu", type=str, default="40GiB",
                        help="Maximum memory per GPU")
    
    # Data configuration
    parser.add_argument("--metadata-file", type=str, 
                        default="./eval/gen/rise/data/datav2_total_w_subtask.json",
                        help="RISE metadata file path")
    parser.add_argument("--rise-data-dir", type=str,
                        default="./eval/gen/rise/data",
                        help="RISE data directory")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to evaluate")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Output directory")
    
    # Inference parameters
    parser.add_argument("--cfg-text-scale", type=float, default=4.0,
                        help="CFG text scale")
    parser.add_argument("--cfg-img-scale", type=float, default=1.5,
                        help="CFG image scale")
    parser.add_argument("--num-timesteps", type=int, default=50,
                        help="Number of denoising steps")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    run_ablation_evaluation(args)


if __name__ == "__main__":
    main()
