import os
import json
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

from inferencer import InterleaveInferencer

import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

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
from safetensors.torch import load_file

model_path = "/home/jake0360/projects/def-sreddy/checkpoints/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
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

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

max_mem_per_gpu = "40GiB"

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
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
)

model = model.eval()
print('Model loaded')

# Initialize Inferencer
inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

print("Loading MMLU Dataset...")
dataset = load_dataset("cais/mmlu", "all", split="test")

# MMLU Config
label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
subject_results = defaultdict(lambda: {"correct": 0, "total": 0})

# max think token is 100
inference_hyper = dict(
    max_think_token_n=100,
    do_sample=False,
)

def format_prompt(ex):
    choices_text = f"A. {ex['choices'][0]}\nB. {ex['choices'][1]}\nC. {ex['choices'][2]}\nD. {ex['choices'][3]}"
    user_content = (
        f"Answer the following multiple-choice question.\n\n"
        f"Question: {ex['question']}\n"
        f"{choices_text}\n\n"
        f"Give only the letter of the correct answer (A, B, C, or D)."
    )
    return user_content

print("Starting Evaluation...")

for i, ex in enumerate(tqdm(dataset, desc="Processing")):
    subject = ex["subject"]
    prompt_text = format_prompt(ex)
    
    try:
        output_dict = inferencer(
            image=None,
            text=prompt_text,
            understanding_output=True,
            **inference_hyper
        )
        
        answer = output_dict['text'].strip().upper()
        answer = answer[0] if len(answer) > 0 else ""
        
        correct_label = label_map[ex["answer"]]
        
        if answer == correct_label:
            subject_results[subject]["correct"] += 1
        subject_results[subject]["total"] += 1
        
    except Exception as e:
        print(f"Error: {e}")
        continue

    if i % 100 == 0 and i > 0:
        print("\n=== PER-SUBJECT ACCURACY ===")
        total_acc_sum = 0
        num_subjects = 0

        for subject in sorted(subject_results.keys()):
            stats = subject_results[subject]
            if stats['total'] > 0:
                acc = stats["correct"] / stats["total"]
                total_acc_sum += acc
                num_subjects += 1
                print(f"{subject}: {acc:.2%} ({stats['correct']}/{stats['total']})")

        if num_subjects > 0:
            avg = total_acc_sum / num_subjects
            print("-" * 30)
            print(f"CURRENT MMLU Score: {avg:.2%}")


print("\n=== PER-SUBJECT ACCURACY ===")
total_acc_sum = 0
num_subjects = 0

for subject in sorted(subject_results.keys()):
    stats = subject_results[subject]
    if stats['total'] > 0:
        acc = stats["correct"] / stats["total"]
        total_acc_sum += acc
        num_subjects += 1
        print(f"{subject}: {acc:.2%} ({stats['correct']}/{stats['total']})")

if num_subjects > 0:
    avg = total_acc_sum / num_subjects
    print("-" * 30)
    print(f"Final MMLU Score: {avg:.2%}")