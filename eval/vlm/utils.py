# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
import yaml
import torch

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae


def load_model_and_tokenizer(args):
    model_path = args.model_path

    # ----- LLM CONFIG -----
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ----- VIT CONFIG -----
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # ----- VAE CONFIG / WEIGHTS -----
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # ----- BAGEL CONFIG (matches inference.ipynb style) -----
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    # ----- BUILD EMPTY MODEL ON META DEVICE -----
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        # meta=True ensures no real weights yet, matches notebook init
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # ----- TOKENIZER -----
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # ----- DEVICE MAP FOR A100-40GB -----
    # Leave a bit of headroom instead of using full 40GiB
    max_mem_per_gpu = "38GiB"
    max_memory = {i: max_mem_per_gpu for i in range(torch.cuda.device_count())}
    # host RAM limit – adjust if you request more/less in salloc
    max_memory["cpu"] = "80GiB"

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    # (Optional) keep some modules on the same device for stability/perf
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
    ]
    if torch.cuda.device_count() == 1:
        # put them all on the first GPU or fallback to cuda:0
        first_device = next(iter(device_map.values())) if len(device_map) > 0 else "cuda:0"
        for k in list(device_map.keys()):
            if any(k.startswith(m) for m in same_device_modules):
                device_map[k] = first_device

    # ----- LOAD CHECKPOINT WITH ACCELERATE -----
    ckpt_path = os.path.join(model_path, "ema.safetensors")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=ckpt_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/bagel_offload",
    )

    model = model.eval()
    return model, tokenizer, new_token_ids


def build_transform():
    with open("./data/configs/example.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    max_image_size = data_config["vlm_sft"]["image_transform_args"]["max_image_size"]
    min_image_size = data_config["vlm_sft"]["image_transform_args"]["min_image_size"]
    image_stride = data_config["vlm_sft"]["image_transform_args"]["image_stride"]
    max_pixels = data_config["vlm_sft"]["image_transform_args"]["max_pixels"]

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform


def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation
