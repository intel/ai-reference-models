# Optimize AI Model Zoo Workloads with PyTorch* using Docker Containers

This document provides links to step-by-step instructions on how to leverage Model Zoo docker containers to run optimized open-source Deep Learning Training and Inference workloads using PyTorch* framework on [4th Generation Intel® Xeon® Scalable processors](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx).

## Use cases

The tables below provide links to run each use case using docker containers. The model scripts run on Linux. 

### Image Recognition

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf) | Training | [FP32 AVX-FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/pytorch/resnet50/training/cpu/README_SPR_DEV_CAT.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf) | Inference | [FP32 INT8 ](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/pytorch/resnet50/inference/cpu/README_SPR_DEV_CAT.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNext-32x16d](https://arxiv.org/pdf/1611.05431.pdf) | Inference | [FP32 AVX-FP32 INT8 AVX-INT8,BF16](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/README_SPR_DEV_CAT.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md)  |

### Object Detection

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) | Training | [FP32 AVX-FP32 BF16 BF32](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/pytorch/maskrcnn/training/cpu/README_SPR_DEV_CAT.md) | [COCO 2017 training dataset](https://github.com/IntelAI/models/tree/master/datasets/coco/README_train.md) |
| [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) | Inference | [FP32 AVX-FP32 BF16 BF32](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/pytorch/maskrcnn/inference/cpu/README_SPR_DEV_CAT.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images) |
| [SSD-ResNet34](https://arxiv.org/abs/1512.02325.pdf) | Training | [FP32 AVX-FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/README_SPR_DEV_CAT.md) | [COCO 2017](/quickstart/object_detection/pytorch/ssd-resnet34/training/cpu/README.md#datasets) |
| [SSD-ResNet34](https://arxiv.org/abs/1512.02325.pdf) | Inference | [FP32 AVX-FP32 INT8 AVX-INT8 BF16](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/README_SPR_DEV_CAT.md) | [COCO 2017](/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/README.md#datasets) |

### Language Modeling 

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) | Training | [FP32 AVX-FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/pytorch/bert_large/training/cpu/README_SPR_DEV_CAT.md) | [Preprocessed Text ataset](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) | Inference | [FP32 AVX-FP32 INT8 AVX-INT8 BF16](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/pytorch/bert_large/inference/cpu/README_SPR_DEV_CAT.md) | [SQuAD1.0](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering) |
| [RNN-T](https://arxiv.org/abs/2007.15188.pdf) | Inference | [FP32 AVX-FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/pytorch/rnnt/inference/cpu/README_SPR_DEV_CAT.md) | [RNN-T dataset](/quickstart/language_modeling/pytorch/rnnt/inference/cpu/download_dataset.sh) |
| [DistilBERT base](https://arxiv.org/abs/1910.01108.pdf) | Inference | [FP32 BF32 BF16 INT8-FP32 INT8-BF16](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/pytorch/distilbert_base/inference/cpu/README_SPR_DEV_CAT.md) | [ DistilBERT Base SQuAD1.1](https://huggingface.co/distilbert-base-uncased-distilled-squad) |

### Recommendation 

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [DLRM](https://arxiv.org/pdf/1906.00091.pdf) | Training | [FP32 AVX-FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/recommendation/pytorch/dlrm/training/cpu/README_SPR_DEV_CAT.md) | [Criteo Terabyte](/quickstart/recommendation/pytorch/dlrm/inference/cpu/README.md#datasets) |
| [DLRM](https://arxiv.org/pdf/1906.00091.pdf) | Inference | [FP32 AVX-FP32 INT8 AVX-INT8 BF16](https://github.com/IntelAI/models/blob/master/quickstart/recommendation/pytorch/dlrm/inference/cpu/README_SPR_DEV_CAT.md) | [Criteo Terabyte](/quickstart/recommendation/pytorch/dlrm/inference/cpu/README.md#datasets) |
