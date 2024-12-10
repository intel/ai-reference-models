# Optimized Intel® Reference Models for Intel® Data Center GPU Max Series

This document provides links to step-by-step instructions on how to leverage reference model docker containers to run optimized open-source Deep Learning training and inference workloads using Intel® Extension for PyTorch* and Intel® Extension for TensorFlow* on the [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html).

## Base Containers

| AI Framework                 | Extension            | Documentation |
| -----------------------------| ------------- | ----------------- |
| PyTorch | Intel® Extension for PyTorch* | [Intel® Extension for PyTorch Container](https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.10%2Bxpu/docker/README.md) |
| TensorFlow | Intel® Extension for TensorFlow* | [Intel® Extension for TensorFlow Container](https://github.com/intel/intel-extension-for-tensorflow/blob/v2.14.0.1/docker/README.md)|

## Optimized Workloads

The table below provides links to run each workload in a docker container. The containers were validated on a host running Linux*.

| Model                            | Framework                  | Mode | Precisions |
| ----------------------------|     ---------- | ------------------- | ------------ |
| [3D-UNet](https://arxiv.org/abs/1606.06650) | TensorFlow | [Training](../../models_v2/tensorflow/3d_unet/training/gpu/CONTAINER.md) | BF16 |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | PyTorch | [Inference](../../models_v2/pytorch/bert_large/inference/gpu/CONTAINER.md) | FP16, BF16 and FP32 |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | PyTorch | [Training](../../models_v2/pytorch/bert_large/training/gpu/CONTAINER.md) | BF16,TF32 and FP32 |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | TensorFlow | [Training](../../models_v2/tensorflow/bert_large/training/gpu/CONTAINER.md) | BF16 |
| [DistilBERT](https://arxiv.org/abs/1910.01108) | PyTorch | [Inference](../../models_v2/pytorch/distilbert/inference/gpu/CONTAINER_MAX.md) | FP16,BF16 and FP32 |
| [DLRM](https://arxiv.org/abs/1906.00091) | PyTorch | [Inference](../../models_v2/pytorch/torchrec_dlrm/inference/gpu/DEVCATALOG.md) | FP16 |
| [DLRM](https://arxiv.org/abs/1906.00091) | PyTorch | [Training](../../models_v2/pytorch/torchrec_dlrm/training/gpu/CONTAINER.md) | FP32,TF32 and BF16 |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870) | TensorFlow | [Training](../../models_v2/tensorflow/maskrcnn/training/gpu/CONTAINER.md) | BF16 |
| [ResNet50 v1.5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | [Inference](../../models_v2/pytorch/resnet50v1_5/inference/gpu/CONTAINER_MAX.md) | INT8,FP16,BF16,FP32 and TF32 |
| [ResNet50 v1.5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | [Training](../../models_v2/pytorch/resnet50v1_5/training/gpu/CONTAINER.md) | BF16,FP32 and TF32 |
| [ResNet50 v1.5](https://arxiv.org/pdf/1512.03385.pdf) | TensorFlow | [Training](../../models_v2/tensorflow/resnet50v1_5/training/gpu/CONTAINER.md) |  BF16 |
| [RNN-T](https://arxiv.org/abs/1211.3711) | PyTorch | [Inference](../../models_v2/pytorch/rnnt/inference/gpu/CONTAINER.md) |
| [RNN-T](https://arxiv.org/abs/1211.3711) | PyTorch | [Training](../../models_v2/pytorch/rnnt/training/gpu/CONTAINER.md) |
| [Stable Diffusion](https://arxiv.org/abs/2112.10752) | PyTorch | [Inference](../../models_v2/pytorch/stable_diffusion/inference/gpu/CONTAINER_MAX.md) | FP16 |

**Note**:
* DLRM(PyTorch) inference, BERT-Large(TensorFlow) inference and ResNet50v1.5(TensorFlow) inference workloads are supported on older Intel® Extension for TensorFlow* v2.13 and Intel® Extension for PyTorch* 2.0.110+xpu versions.
* The other models in the list are validated on Intel® Extension for TensorFlow* v2.14 and Intel® Extension for PyTorch* 2.1.10+xpu versions.
