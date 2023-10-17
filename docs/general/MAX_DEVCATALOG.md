# Model Zoo for Intel® Architecture Workloads Optimized for the Intel® Data Center GPU Max Series

This document provides links to step-by-step instructions on how to leverage Model Zoo docker containers to run optimized open-source Deep Learning training and inference workloads using Intel® Extension for PyTorch* and Intel® Extension for TensorFlow* on the [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html).

## Base Containers

| AI Framework                 | Extension            | Documentation |
| -----------------------------| ------------- | ----------------- |
| PyTorch | Intel® Extension for PyTorch* | [Intel® Extension for PyTorch Container](https://github.com/intel/intel-extension-for-pytorch/blob/v2.0.110%2Bxpu/docker/README.md) |
| TensorFlow | Intel® Extension for TensorFlow* | [Intel® Extension for TensorFlow Container](https://github.com/intel/intel-extension-for-tensorflow/blob/v2.13.0.0/docker/README.md)|

## Optimized Workloads

The table below provides links to run each workload in a docker container. The containers are optimized for Linux*. 

| Model                            | Framework                  | Mode | Precisions | Dataset |
| ----------------------------|     ---------- | ------------------- | ------------ | ------------ |
| [ResNet50v1-5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | [Inference](https://github.com/IntelAI/models/blob/v3.0.0/quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/DEVCATALOG_MAX.md) | INT8 | [ImageNet 2012](https://github.com/IntelAI/models/tree/v3.0.0/datasets/imagenet/README.md) |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | PyTorch | [Inference](https://github.com/IntelAI/models/blob/v3.0.0/quickstart/language_modeling/pytorch/bert_large/inference/gpu/DEVCATALOG.md) | FP16 | [SQuAD](https://github.com/IntelAI/models/blob/v3.0.0/datasets/bert_data/README.md#inference) |
| [DLRM](https://arxiv.org/abs/1906.00091) | PyTorch | [Inference](https://github.com/IntelAI/models/blob/v3.0.0/quickstart/recommendation/pytorch/torchrec_dlrm/inference/gpu/DEVCATALOG.md) | FP16 | [Criteo](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf) |
| [DLRM](https://arxiv.org/abs/1906.00091) | PyTorch | [Training](https://github.com/IntelAI/models/blob/v3.0.0/quickstart/recommendation/pytorch/torchrec_dlrm/training/gpu/DEVCATALOG.md) | BF16 | [Criteo](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf) | 
| [ResNet50v1-5](https://arxiv.org/pdf/1512.03385.pdf) | TensorFlow| [Inference](https://github.com/IntelAI/models/blob/v3.0.0/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/gpu/DEVCATALOG_MAX.md) | INT8,FP16,FP32 | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | TensorFlow | [Inference](https://github.com/IntelAI/models/blob/v3.0.0/quickstart/language_modeling/tensorflow/bert_large/inference/gpu/DEVCATALOG.md) | FP16,FP32 | [SQuAD ](https://github.com/IntelAI/models/blob/v3.0.0/datasets/bert_data/README.md#inference) |
| [ResNet50v1-5](https://arxiv.org/pdf/1512.03385.pdf) | TensorFlow | [Training](https://github.com/IntelAI/models/blob/r2.12/quickstart/image_recognition/tensorflow/resnet50v1_5/training/gpu/DEVCATALOG.md) |  BF16 | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet50v1-5](https://arxiv.org/pdf/1512.03385.pdf) | PyTorch | [Training](https://github.com/IntelAI/models/blob/r2.12/quickstart/image_recognition/pytorch/resnet50v1_5/training/gpu/DEVCATALOG.md) | BF16 |  [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | TensorFlow | [Training](https://github.com/IntelAI/models/blob/r2.12/quickstart/language_modeling/tensorflow/bert_large/training/gpu/DEVCATALOG.md) | BF16 | Dummy dataset |
| [BERT Large](https://arxiv.org/pdf/1810.04805.pdf)                                           | PyTorch | [Training](https://github.com/IntelAI/models/blob/r2.12/quickstart/language_modeling/pytorch/bert_large/training/gpu/DEVCATALOG.md) | BF16 | [MLCommons](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/pytorch/bert_large/training/gpu/DEVCATALOG.md#datasets) |


**Note**: ResNet50v1.5 and BERT-Large training workloads are supported on older Intel® Extension for TensorFlow* v2.12 and Intel® Extension for PyTorch* 1.13.100+xpu versions. The other models in the list are validated on Intel® Extension for TensorFlow* v2.13 and Intel® Extension for PyTorch* 2.0.100+xpu versions.
