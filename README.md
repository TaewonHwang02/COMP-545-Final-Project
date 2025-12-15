# COMP 545 Final Project

This repository is cloned from the [BAGEL](https://github.com/bytedance-seed/BAGEL) repository. We analyze the interal architecture and mechanism of the `BAGEL` model by performing various ablation studies.

## 🔥 Quick Start

1️⃣ Set up environment

```bash
git clone https://github.com/bytedance-seed/BAGEL.git
cd BAGEL
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

2️⃣ Download pretrained checkpoint

```python
from huggingface_hub import snapshot_download

save_dir = "models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

```

## Experiments

- Weight modification analysis is performed using `weight_modification_experiment.ipynb` file.
- Language reasoning capability comparison between BAGEL model and Qwen2.5 model is done using `qwen_reasoning_mmlu.py` & `bagel_reasoning_mmlu.py`
- `benchmark.sh` file is used to utilize slurm on Compute Canada server to run experiments.

## 📜 License

BAGEL is licensed under the Apache 2.0.
