# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Ablation Study: Removing ViT Tokens
Test image-to-image generation capability using only VAE/Generation Expert
"""

import os
from copy import deepcopy
import random
import numpy as np

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 80)
print("Ablation Study: Removing ViT Tokens - Using VAE/Generation Expert Only")
print("=" * 80)

# Model path (adjust to your checkpoint location)
model_path = "./checkpoints/BAGEL-7B-MoT"
print(f"Model path: {model_path}")

# LLM Configuration
print("\nPreparing configuration files...")
print("  - Loading LLM config...")
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"
print("    LLM config loaded successfully")

# ViT Configuration
print("  - Loading ViT config...")
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
print("    ViT config loaded successfully")

# VAE Loading
print("\n[Step 1/8] Loading VAE model...")
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
print("VAE model loaded successfully")

# Bagel Configuration
print("\n[Step 2/8] Preparing Bagel configuration...")
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
print("Bagel config created successfully")

# Initialize Model
print("\n[Step 3/8] Initializing model structure...")
with init_empty_weights():
    print("  - Initializing language model...")
    language_model = Qwen2ForCausalLM(llm_config)
    print("  - Initializing vision model...")
    vit_model      = SiglipVisionModel(vit_config)
    print("  - Initializing Bagel model...")
    model          = Bagel(language_model, vit_model, config)
    print("  - Converting ViT conv2d layers...")
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
print("Model structure initialized successfully")

# Tokenizer Preparation
print("\n[Step 4/8] Loading Tokenizer...")
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
print("Tokenizer loaded successfully")

# Image Transform Preparation
print("\n[Step 5/8] Preparing image transforms...")
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)
print("Image transforms prepared successfully")

# Model Loading and Multi-GPU Setup
print("\n[Step 6/8] Setting up device mapping...")
max_mem_per_gpu = "40GiB"  # Adjust based on your GPU
print(f"  - Detected {torch.cuda.device_count()} GPU(s)")
print(f"  - Max memory per GPU: {max_mem_per_gpu}")

print("  - Inferring device mapping...")
device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print("Device mapping completed")

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
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

print("\n[Step 7/8] Loading model weights...")
print("  - This may take a few minutes, please wait...")
checkpoint_path = os.path.join(model_path, "ema.safetensors")
print(f"  - Weights file: {checkpoint_path}")
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=checkpoint_path,
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
)

print("  - Setting model to evaluation mode...")
model = model.eval()
print("Model loaded successfully!")

# Prepare Inferencer
print("\n[Step 8/8] Preparing inferencer...")
inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)
print("Inferencer initialized successfully")
print("\n" + "=" * 80)
print("All preparations completed! Starting image generation experiment...")
print("=" * 80)

# Inference hyperparameters
inference_hyper = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.5,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
)

# Load input image
print("\n" + "=" * 80)
print("Ablation Study: Image-to-Image - Maze Solving Task")
print("=" * 80)

input_image_path = "./eval/gen/rise/data/data/logical_reasoning_images/53.png"
print(f"Loading input image: {input_image_path}")

try:
    input_image = Image.open(input_image_path).convert('RGB')
    print(f"Image loaded successfully, size: {input_image.size}")
except Exception as e:
    print(f"Failed to load image: {e}")
    exit(1)

# Resize image to fit VAE
input_image = vae_transform.resize_transform(pil_img2rgb(input_image))
print(f"Resized image size: {input_image.size}")

# Initialize generation context
print("\nStarting inference (without ViT tokens)...")
print("-" * 80)
print("Experiment setup:")
print("  VAE encoding: Enabled (provides pixel-level details)")
print("  ViT encoding: Disabled (removing semantic understanding tokens)")
print("  Generation Expert: Enabled")
print("-" * 80)

with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
    # Initialize context
    gen_context = inferencer.init_gen_context()
    cfg_text_context = deepcopy(gen_context)
    cfg_img_context = deepcopy(gen_context)
    
    # Step 1: Add input image (VAE only, no ViT)
    print("\nStep 1: Encoding input image (VAE only)...")
    gen_context = inferencer.update_context_image(
        input_image, 
        gen_context, 
        vae=True,   # Enable VAE encoding
        vit=False   # Disable ViT encoding (key for ablation)
    )
    print(f"  Current sequence length: {gen_context['kv_lens']}")
    
    # Create cfg_text context checkpoint
    cfg_text_context = deepcopy(gen_context)
    
    # Step 2: Add text prompt
    prompt = ("The image shows a maze. \nRules:\n"
              "1. The maze consists of a grid of cells\n"
              "2. Walls are represented by black line between cells, not as cells themselves\n"
              "3. You can move horizontally or vertically between adjacent cells if there is no wall between them\n"
              "4. The goal is to find a path from the start cell (S) to the end cell (E)\n\n"
              "Please draw the solution path with a red line.")
    print(f"\nStep 2: Adding text prompt...")
    print(f"  Prompt: {prompt[:80]}...")
    
    gen_context = inferencer.update_context_text(prompt, gen_context)
    cfg_img_context = inferencer.update_context_text(prompt, cfg_img_context)
    print(f"  Current sequence length: {gen_context['kv_lens']}")
    
    # Step 3: Generate image
    print(f"\nStep 3: Generating image (without ViT)...")
    print(f"  Denoising steps: {inference_hyper['num_timesteps']}")
    print(f"  CFG text scale: {inference_hyper['cfg_text_scale']}")
    print(f"  CFG image scale: {inference_hyper['cfg_img_scale']}")
    print(f"  Estimated time: ~{inference_hyper['num_timesteps'] * 2} seconds")
    print("  Generating, please wait...")
    
    image_shape = input_image.size[::-1]  # (H, W)
    output_image = inferencer.gen_image(
        image_shape, 
        gen_context, 
        cfg_text_precontext=cfg_text_context,
        cfg_img_precontext=cfg_img_context,
        **inference_hyper
    )
    print("  Generation completed!")

# Save results
output_path_no_vit = "output_ablation_no_vit_evaluation.png"
output_image.save(output_path_no_vit)
print(f"\nOutput image saved: {output_path_no_vit}")

# Save a copy of input image for comparison
input_copy_path = "evaluation_copy.jpg"
input_image.save(input_copy_path)
print(f"Input image copy saved: {input_copy_path}")

#############################################################################
# Comparison Experiment: Image Generation with ViT + VAE
#############################################################################
print("\n\n" + "=" * 80)
print("Comparison Experiment: Image-to-Image - Full Model (VAE + ViT)")
print("=" * 80)

print("\nStarting inference (with ViT tokens)...")
print("-" * 80)
print("Experiment setup:")
print("  VAE encoding: Enabled (provides pixel-level details)")
print("  ViT encoding: Enabled (provides semantic understanding tokens)")
print("  Generation Expert: Enabled")
print("-" * 80)

with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
    # Initialize context
    print("\nInitializing new generation context...")
    gen_context_with_vit = inferencer.init_gen_context()
    cfg_text_context_with_vit = deepcopy(gen_context_with_vit)
    cfg_img_context_with_vit = deepcopy(gen_context_with_vit)
    
    # Step 1: Add input image (VAE + ViT)
    print("\nStep 1: Encoding input image (VAE + ViT)...")
    gen_context_with_vit = inferencer.update_context_image(
        input_image, 
        gen_context_with_vit, 
        vae=True,   # Enable VAE encoding
        vit=True    # Enable ViT encoding (full model)
    )
    print(f"  Current sequence length: {gen_context_with_vit['kv_lens']}")
    print("  Note: Sequence length is longer than no-ViT version (includes ViT tokens)")
    
    # Create cfg_text context
    cfg_text_context_with_vit = deepcopy(gen_context_with_vit)
    
    # Step 2: Add text prompt
    print(f"\nStep 2: Adding text prompt...")
    print(f"  Prompt: {prompt[:80]}...")
    
    gen_context_with_vit = inferencer.update_context_text(prompt, gen_context_with_vit)
    cfg_img_context_with_vit = inferencer.update_context_text(prompt, cfg_img_context_with_vit)
    print(f"  Current sequence length: {gen_context_with_vit['kv_lens']}")
    
    # Step 3: Generate image
    print(f"\nStep 3: Generating image (with full model VAE+ViT)...")
    print(f"  Denoising steps: {inference_hyper['num_timesteps']}")
    print(f"  CFG text scale: {inference_hyper['cfg_text_scale']}")
    print(f"  CFG image scale: {inference_hyper['cfg_img_scale']}")
    print(f"  Estimated time: ~{inference_hyper['num_timesteps'] * 2} seconds")
    print("  Generating, please wait...")
    
    output_image_with_vit = inferencer.gen_image(
        image_shape, 
        gen_context_with_vit, 
        cfg_text_precontext=cfg_text_context_with_vit,
        cfg_img_precontext=cfg_img_context_with_vit,
        **inference_hyper
    )
    print("  Generation completed!")

# Save with-ViT results
output_path_with_vit = "output_ablation_with_vit_eval.png"
output_image_with_vit.save(output_path_with_vit)
print(f"\nOutput image saved: {output_path_with_vit}")

print("\n" + "=" * 80)
print("Ablation Comparison Experiment Completed!")
print("=" * 80)
print(f"""
Experiment Summary - Ablation Comparison:
  
Input:
  - Input image: {input_image_path}
  - Copy saved: {input_copy_path}
  
Output:
  - No ViT version: {output_path_no_vit}
  - With ViT version: {output_path_with_vit}
  
Experiment Settings:
  - Task type: Image-to-Image - Maze Solving
  - Prompt: "{prompt[:50]}..."
  
Comparison Analysis:
  
  Experiment A (No ViT):
    VAE encoding: Enabled (pixel-level features)
    ViT encoding: Disabled
    -> Tests pure pixel-level generation capability
  
  Experiment B (With ViT):
    VAE encoding: Enabled (pixel-level features)
    ViT encoding: Enabled (semantic features)
    -> Tests full model generation capability
  
Research Value:
  By comparing these two output images, we can quantitatively analyze:
  1. ViT tokens' contribution to semantic understanding and task completion
  2. VAE tokens' independent role in image generation
  3. Importance of multimodal feature fusion
  4. Generation Expert's performance under different input conditions
  
Suggestions:
  Open both output images for visual comparison, observe:
  - Task completion accuracy
  - Detail preservation
  - Overall image quality
  - Semantic consistency
""")
