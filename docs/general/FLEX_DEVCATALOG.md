# Optimized Intel® Reference Models for the Intel® Data Center GPU Flex Series

This document provides links to step-by-step instructions on how to leverage reference model docker containers to run optimized open-source Deep Learning inference workloads using Intel® Extension for PyTorch* and Intel® Extension for TensorFlow* on the [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html).

## Base Containers

| AI Framework                 | Extension            | Documentation |
| -----------------------------| ------------- | ----------------- |
| PyTorch | Intel® Extension for PyTorch* | [Intel® Extension for PyTorch Container](https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.10%2Bxpu/docker/README.md) |
| TensorFlow | Intel® Extension for TensorFlow* | [Intel® Extension for TensorFlow Container](https://github.com/intel/intel-extension-for-tensorflow/blob/v2.14.0.1/docker/README.md)|

## Optimized Workloads

The table below provides links to run each workload in a docker container. The containers were validated on a host running Linux*.


| Model                            | Framework                  | Mode and Documentation     |
| ----------------------------|     ---------- | ----------|
| [DistilBERT](https://arxiv.org/abs/1910.01108) | PyTorch | [FP16 and FP32  Inference](../../models_v2/pytorch/distilbert/inference/gpu/CONTAINER_FLEX.md) |
| [DLRM v1](https://arxiv.org/abs/1906.00091) | PyTorch | [ FP16 Inference](../../models_v2/pytorch/dlrm/inference/gpu/CONTAINER.md) |
| [EfficientNet](https://arxiv.org/abs/1905.11946) B0,B3,B4 | TensorFlow | [FP16 Inference](../../models_v2/tensorflow/efficientnet/inference/gpu/CONTAINER.md) |
| [EfficientNet](https://arxiv.org/abs/1905.11946) | PyTorch | [FP32 FP16 BF16 Inference](../../models_v2/pytorch/efficientnet/inference/gpu/README.md)
| [HiFi-GAN](https://arxiv.org/abs/2010.05646) | PyTorch | [FP16 Inference](../../models_v2/pytorch/hifi_gan/inference/gpu/CONTAINER.md) |
| [FastPitch](https://arxiv.org/abs/2006.06873) | PyTorch | [FP16 Inference](../../models_v2/pytorch/fastpitch/inference/gpu/CONTAINER.md) |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870) | TensorFlow | [FP16 Inference](../../models_v2/tensorflow/maskrcnn/inference/gpu/CONTAINER.md) |
| [QuartzNet](https://arxiv.org/abs/1910.10261) | PyTorch | [FP16 Inference](../../models_v2/pytorch/quartznet/inference/gpu/CONTAINER.md) |
| [ResNet50 v1.5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | [INT8 Inference](../../models_v2/pytorch//resnet50v1_5/inference/gpu/CONTAINER_FLEX.md) |
| [ResNet50 v1.5](https://arxiv.org/pdf/1512.03385.pdf) | TensorFlow | [INT8 Inference](../../models_v2/tensorflow/resnet50v1_5/inference/gpu/CONTAINER_FLEX.md) |
| [SSD-MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf) | TensorFlow | [INT8 Inference](../../models_v2/tensorflow/ssd-mobilenet/inference/gpu/DEVCATALOG.md) |
| [SSD-MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf) | PyTorch | [INT8 Inference](../../models_v2/pytorch/ssd-mobilenet/inference/gpu/DEVCATALOG.md) |
| [Stable Diffusion](https://arxiv.org/abs/2112.10752) | PyTorch | [Flex GPU FP16 Inference](../../models_v2/pytorch/stable_diffusion/inference/gpu/CONTAINER_FLEX.md) and [Arc GPU FP16 Inference](../../models_v2/pytorch/stable_diffusion/inference/gpu/WSL2_ARC_STABLE_DIFFUSION.md) |
| [Stable Diffusion](https://arxiv.org/abs/2112.10752) | TensorFlow | [FP32,FP16 Inference](../../models_v2/tensorflow/stable_diffusion/inference/gpu/CONTAINER.md) |
| [UNet++](https://arxiv.org/abs/1807.10165) | PyTorch | [FP16 Inference](../../models_v2/pytorch/unetpp/inference/gpu/CONTAINER.md) |
| [Swin Transformer](https://arxiv.org/abs/2103.14030) | PyTorch | [FP16 Inference](../../models_v2/pytorch/swin-transformer/inference/gpu/CONTAINER.md) |
| [Wide and Deep](https://arxiv.org/abs/1606.07792) | TensorFlow | [FP16 Inference](../../models_v2/tensorflow/wide_deep_large_ds/inference/gpu/CONTAINER.md) |
| [YOLO v4](https://arxiv.org/pdf/1704.04861.pdf) | PyTorch | [INT8 Inference](../../models_v2/pytorch/yolov4/inference/gpu/DEVCATALOG.md) |
| [YOLO v5](https://ui.adsabs.harvard.edu/abs/2021zndo...4679653J/abstract) | PyTorch | [FP16 Inference](../../models_v2/pytorch/yolov5/inference/gpu/README.md) |


**Note**:
* SSD-MobileNet v1, and YOLO v4 models are supported on older Intel® Extension for TensorFlow* v2.12 and Intel® Extension for PyTorch* 1.13.120+xpu versions.
* Stable Diffusion for Arc GPU was validated on PyTorch* 2.0.110+xpu version.
* The other models in the list are validated on Intel® Extension for TensorFlow* v2.14 and Intel® Extension for PyTorch* 2.1.10+xpu versions.
