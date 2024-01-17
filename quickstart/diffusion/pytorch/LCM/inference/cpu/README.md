<!--- 0. Title -->
# PyTorch Latent Consistency Models inference

<!-- 10. Description -->
## Description

This document has instructions for running [Latent Consistency Models (LCMs).](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Install dependencies
```
pip install torchmetrics pycocotools transformers==4.35.2 peft==0.6.2
pip install torch-fidelity --no-deps
```

### install model
```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.23.1
git apply ../diffusers.patch
python setup.py install
```

* [optional] Compile model with PyTorch Inductor backend
```shell
export TORCH_INDUCTOR=1
```

# Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash inference_throughput.sh fp32 ipex-jit | bash inference_realtime.sh fp32 ipex-jit | bash accuracy.sh fp32 ipex-jit |
| BF16        | bash inference_throughput.sh bf16 ipex-jit | bash inference_realtime.sh bf16 ipex-jit | bash accuracy.sh bf16 ipex-jit |
| BF32        | bash inference_throughput.sh bf32 ipex-jit | bash inference_realtime.sh bf32 ipex-jit | bash accuracy.sh bf32 ipex-jit |
| FP16        | bash inference_throughput.sh fp16 ipex-jit | bash inference_realtime.sh fp16 ipex-jit | bash accuracy.sh fp16 ipex-jit |


## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variable set to point to an output directory.

```bash
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)
cd quickstart/diffusion/pytorch/stable_diffusion/inference/cpu/

# Env vars
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, FP32 multi-instance realtime inference)
bash inference_realtime.sh fp32 ipex-jit
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
