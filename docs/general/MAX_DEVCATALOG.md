# Model Zoo for Intel® Architecture Workloads Optimized for the Intel® Data Center GPU Max Series

This document provides links to step-by-step instructions on how to leverage Model Zoo docker containers to run optimized open-source Deep Learning training and inference workloads using Intel® Extension for PyTorch* and Intel® Extension for TensorFlow* on the [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html).

## Base Containers

| AI Framework                 | Extension            | Documentation |
| -----------------------------| ------------- | ----------------- |
| PyTorch | Intel® Extension for PyTorch* | [Intel® Extension for PyTorch Container](https://github.com/intel/intel-extension-for-pytorch/blob/xpu-master/docker/README.md) |
| TensorFlow | Intel® Extension for TensorFlow* | [Intel® Extension for TensorFlow Container](https://github.com/intel/intel-extension-for-tensorflow/blob/r1.1/docker/README.md)|

## Optimized Workloads

The table below provides links to run each workload in a docker container. The containers are optimized for Linux*. 

| Model                            | Framework                  | Mode | Precisions | Dataset |
| ----------------------------|     ---------- | ------------------- | ------------ | ------------ |
| [ResNet50v1-5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | [Inference](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/DEVCATALOG_MAX.md) | INT8 | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet50v1-5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | [Training](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/pytorch/resnet50v1_5/training/gpu/DEVCATALOG.md) | BF16 |  [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | PyTorch | [Inference](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/pytorch/bert_large/inference/gpu/DEVCATALOG.md) | FP16 | [SQuAD](https://github.com/IntelAI/models/blob/master/datasets/bert_data/README.md#inference) |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | PyTorch | [Training](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/pytorch/bert_large/training/gpu/DEVCATALOG.md) | BF16 | [MLCommons](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/pytorch/bert_large/training/gpu/DEVCATALOG.md#datasets) |
| [ResNet50v1-5](https://arxiv.org/pdf/1512.03385.pdf) | TensorFlow| [Inference](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/DEVCATALOG_MAX.md) | INT8,FP16,FP32 | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet50v1-5](https://arxiv.org/pdf/1512.03385.pdf) | TensorFlow | [Training](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/tensorflow/resnet50v1_5/training/gpu/DEVCATALOG.md) |  BF16 | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | TensorFlow | [Inference](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/tensorflow/bert_large/inference/gpu/DEVCATALOG.md) | FP16,FP32 | [SQuAD ](https://github.com/IntelAI/models/blob/master/datasets/bert_data/README.md#inference) |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | TensorFlow | [Training](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/tensorflow/bert_large/training/gpu/DEVCATALOG.md) | BF16 | Dummy dataset |
