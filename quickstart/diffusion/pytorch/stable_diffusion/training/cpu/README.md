<!--- 0. Title -->
# PyTorch Stable Diffusion fine-tuning

<!-- 10. Description -->
## Description

This document has instructions for running [Stable Diffusion, which is a text-to-image latent diffusion model created by the researchers and engineers from CompVis, Stability AI, LAION and RunwayML.](https://huggingface.co/runwayml/stable-diffusion-v1-5) fine-tuning using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Install dependencies
```bash
pip install torchmetrics torch-fidelity pycocotools transformers==4.28.1
```

### install model
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.16.0
python setup.py install
```

### download dataset
We will use some cat images: https://huggingface.co/datasets/diffusers/cat_toy_example .
Let's first download it locally:
```bash
python download_dataset.py
```

### Install Stable Diffusion fine-tuning dependencies
```bash
pip install -r requirements.txt
```

### Initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:
```bash
accelerate config
```

### Model Specific Setup
* Set Jemalloc Preload for better performance
The tcmalloc should be built from the [General setup](#general-setup) section.
```bash
    export LD_PRELOAD="path/lib/libtcmalloc.so":$LD_PRELOAD
```

* Set IOMP preload for better performance
IOMP should be installed in your conda env from the [General setup](#general-setup) section.
```bash
    export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use fp16 AMX if you are using a supported platform
```bash
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `finetune.sh` | Fine-tuning using one socket for the specified precision (fp32, bf16, bf32 or fp16). |
| `finetune_dist.sh` | Distributed fine-tuning for the specified precision (fp32, bf16, bf32 or fp16). |


## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variable set to point to an output directory.

```bash
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)
# Install diffusers
cd quickstart/diffusion/pytorch/stable_diffusion/training/cpu
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.16.0
python setup.py install
cd ..
# Run fine-tuning
bash finetune.sh
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
