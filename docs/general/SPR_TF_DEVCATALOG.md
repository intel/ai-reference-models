# Optimize AI Model Zoo Workloads with TensorFlow using Docker Containers

This document provides links to step-by-step instructions on how to leverage Model Zoo docker containers to run optimized open-source Deep Learning Training and Inference workloads using TensorFlow framework on [4th Generation Intel® Xeon® Scalable processors](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx).

## Use cases

The tables below provide links to run each use case using docker containers. The model scripts run on Linux. 

### Image Recognition

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet) | Training | [FP32 BF16 BF32](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/README_SPR_DEV_CAT.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet) | Inference | [FP32 INT8 BF16 BF32](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/README_SPR_DEV_CAT.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [MobileNet V1*](https://arxiv.org/pdf/1704.04861.pdf) | Inference | [FP32 INT8 BF16 BF32](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/README_SPR_DEV_CAT.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |

### Image Segmentation

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [3D U-Net MLPerf*](https://arxiv.org/pdf/1606.06650.pdf) | Inference | [FP32 BF16 INT8](https://github.com/IntelAI/models/blob/master/quickstart/image_segmentation/tensorflow/3d_unet_mlperf/inference/cpu/README_SPR_DEV_CAT.md) | [BRATS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) |

### Object Detection 

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [SSD-ResNet34](https://arxiv.org/pdf/1512.02325.pdf) | Training | [FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/README_SPR_DEV_CAT.md) | [COCO 2017 training dataset](https://github.com/IntelAI/models/tree/master/datasets/coco/README_train.md) |
| [SSD-ResNet34](https://arxiv.org/pdf/1512.02325.pdf) | Inference | [FP32 INT8 BF16](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/README_SPR_DEV_CAT.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images) |
| [SSD-MobileNet*](https://arxiv.org/pdf/1704.04861.pdf) | Inference | [FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/tensorflow/ssd-mobilenet/inference/cpu/README_SPR_DEV_CAT.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images) |

### Language Modeling 

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) | Training | [FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/tensorflow/bert_large/training/cpu/README_SPR_DEV_CAT.md) |  [SQuAD](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#fine-tuning-with-bert-using-squad-data) and [MRPC](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#classification-training-with-bert) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) | Inference | [FP32 INT8 BF16](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/README_SPR_DEV_CAT.md) | [SQuAD](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#inference) |

### Language Translation 

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
| [Transformer_LT_mlperf*](https://arxiv.org/pdf/1706.03762.pdf) | Training | [FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/README_SPR_DEV_CAT.md) | [WMT English-German dataset](https://github.com/IntelAI/models/tree/master/datasets/transformer_data#transformer-language-mlperf-dataset) |
|  [Transformer_LT_mlperf*](https://arxiv.org/pdf/1706.03762.pdf) | Inference | [FP32 BF16](https://github.com/IntelAI/models/blob/master/quickstart/language_translation/tensorflow/transformer_mlperf/inference/cpu/README_SPR_DEV_CAT.md) |  [WMT English-German dataset](https://github.com/IntelAI/models/tree/master/datasets/transformer_data#transformer-language-mlperf-dataset) |

### Recommendation

| Model                                                  | Mode      | Model Documentation |  Dataset |
| ------------------------------------------------------ | ----------| ------------------- | ---------------------- |
|[DIEN](https://arxiv.org/abs/1809.03672.pdf) | Training | [FP32](https://github.com/IntelAI/models/blob/master/quickstart/recommendation/tensorflow/dien/training/cpu/README_SPR_DEV_CAT.md) | [DIEN dataset](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/dien#1-prepare-datasets-1) |
| [DIEN](https://arxiv.org/abs/1809.03672.pdf) | Inference | [FP32,BF16,BF32](https://github.com/IntelAI/models/blob/master/quickstart/recommendation/tensorflow/dien/inference/cpu/README_SPR_DEV_CAT.md) | [DIEN dataset](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/dien/inference#datasets) |

