<!--- 0. Title -->
# PyTorch Stable Diffusion inference

<!-- 10. Description -->
## Description

This document has instructions for running [Stable Diffusion, which is a text-to-image latent diffusion model created by the researchers and engineers from CompVis, Stability AI, LAION and RunwayML.](https://huggingface.co/CompVis/stable-diffusion-v1-4) inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### install model
```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.16.0
git apply ../diffusers.patch
python setup.py install
```

* Do calibration to get quantization config before running INT8.
```
bash do_calibration.sh
```

# Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash inference_throughput.sh fp32 | bash inference_realtime.sh fp32 | bash accuracy.sh fp32 |
| BF16        | bash inference_throughput.sh bf16 | bash inference_realtime.sh bf16 | bash accuracy.sh bf16 |
| FP16        | bash inference_throughput.sh fp16 | bash inference_realtime.sh fp16 | bash accuracy.sh fp16 |
| INT8        | bash inference_throughput.sh int8 | bash inference_realtime.sh int8 | bash accuracy.sh int8 |

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variable set to point to an output directory.

```
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)
# Clone the Transformers repo in the VIT Base inference directory
cd quickstart/diffusion/pytorch/stable_diffusion/inference/cpu/
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.16.0
git apply ../diffusers.patch
python setup.py install
cd ..
# Env vars
export OUTPUT_DIR=<path to an output directory>
# Run a quickstart script (for example, FP32 multi-instance realtime inference)
bash inference_realtime.sh fp32
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
