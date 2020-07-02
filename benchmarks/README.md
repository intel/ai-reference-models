# Model Zoo Scripts

Training and inference scripts with Intel-optimized MKL

## Prerequisites

The model scripts can be run on Linux and require the following
dependencies to be installed:
* [Docker](https://docs.docker.com/install/)
* [Python](https://www.python.org/downloads/) 3.5 or later
* [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* `wget` for downloading pre-trained models

## TensorFlow Use Cases

| Use Case               | Framework     | Model               | Mode      | Instructions    |
| -----------------------| --------------| ------------------- | --------- |------------------------------|
| Image Recognition      | TensorFlow    | [DenseNet169](https://arxiv.org/pdf/1608.06993.pdf)         | Inference | [FP32](image_recognition/tensorflow/densenet169/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [Inception V3](https://arxiv.org/pdf/1512.00567.pdf)        | Inference | [Int8](image_recognition/tensorflow/inceptionv3/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/inceptionv3/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [Inception V4](https://arxiv.org/pdf/1602.07261.pdf)        | Inference | [Int8](image_recognition/tensorflow/inceptionv4/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/inceptionv4/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [MobileNet V1*](https://arxiv.org/pdf/1704.04861.pdf)        | Inference | [Int8](image_recognition/tensorflow/mobilenet_v1/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/mobilenet_v1/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [ResNet 101](https://arxiv.org/pdf/1512.03385.pdf)          | Inference | [Int8](image_recognition/tensorflow/resnet101/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet101/README.md#fp32-inference-instructions) |
| Image Recognition      | TensorFlow    | [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)           | Inference | [Int8](image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet50/README.md#fp32-inference-instructions)|
| Image Recognition      | TensorFlow    | [ResNet 50v1.5](https://github.com/tensorflow/models/tree/master/official/resnet) | Inference | [Int8](image_recognition/tensorflow/resnet50v1_5/README.md#int8-inference-instructions) [FP32](image_recognition/tensorflow/resnet50v1_5/README.md#fp32-inference-instructions) [BFloat16**](image_recognition/tensorflow/resnet50v1_5/README.md#bfloat16-inference-instructions)|
| Image Recognition      | TensorFlow    | [ResNet 50v1.5](https://github.com/tensorflow/models/tree/master/official/resnet) | Training | [FP32](image_recognition/tensorflow/resnet50v1_5/README.md#fp32-training-instructions) [BFloat16**](image_recognition/tensorflow/resnet50v1_5/README.md#bfloat16-training-instructions)|
| Language Modeling      | TensorFlow    | [BERT](https://arxiv.org/pdf/1810.04805.pdf)                | Inference | [FP32](language_modeling/tensorflow/bert_large/README.md#fp32-inference-instructions) [BFloat16**](language_modeling/tensorflow/bert_large/README.md#bfloat16-inference-instructions) |
| Language Modeling      | TensorFlow    | [BERT](https://arxiv.org/pdf/1810.04805.pdf)                | Training  | [FP32](language_modeling/tensorflow/bert_large/README.md#fp32-training-instructions) [BFloat16**](language_modeling/tensorflow/bert_large/README.md#bfloat16-training-instructions) |
| Language Translation   | TensorFlow    | [GNMT*](https://arxiv.org/pdf/1609.08144.pdf)                | Inference | [FP32](language_translation/tensorflow/mlperf_gnmt/README.md#fp32-inference-instructions) |
| Reinforcement          | TensorFlow    | [MiniGo](https://arxiv.org/abs/1712.01815.pdf)              | Training  | [FP32](reinforcement/tensorflow/minigo/README.md#fp32-training-instructions)|
| Language Translation   | TensorFlow    | [Transformer_LT_Official ](https://arxiv.org/pdf/1706.03762.pdf)| Inference | [FP32](language_translation/tensorflow/transformer_lt_official/README.md#fp32-inference-instructions) |
| Language Translation   | TensorFlow    | [Transformer_LT_mlperf ](https://arxiv.org/pdf/1706.03762.pdf)| Training | [FP32](language_translation/tensorflow/transformer_mlperf/README.md#fp32-training-instructions) [BFloat16**](language_translation/tensorflow/transformer_mlperf/README.md#bfloat16-training-instructions) |
| Object Detection       | TensorFlow    | [R-FCN](https://arxiv.org/pdf/1605.06409.pdf)               | Inference | [Int8](object_detection/tensorflow/rfcn/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/rfcn/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [SSD-MobileNet*](https://arxiv.org/pdf/1704.04861.pdf)       | Inference | [Int8](object_detection/tensorflow/ssd-mobilenet/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/ssd-mobilenet/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [SSD-ResNet34*](https://arxiv.org/pdf/1512.02325.pdf)        | Inference | [Int8](object_detection/tensorflow/ssd-resnet34/README.md#int8-inference-instructions) [FP32](object_detection/tensorflow/ssd-resnet34/README.md#fp32-inference-instructions) |
| Object Detection       | TensorFlow    | [SSD-ResNet34](https://arxiv.org/pdf/1512.02325.pdf)        | Training  | [FP32](object_detection/tensorflow/ssd-resnet34/README.md#fp32-training-instructions) [BFloat16**](object_detection/tensorflow/ssd-resnet34/README.md#bf16-training-instructions) |
| Recommendation         | TensorFlow    | [Wide & Deep Large Dataset](https://arxiv.org/pdf/1606.07792.pdf)	| Inference | [Int8](recommendation/tensorflow/wide_deep_large_ds/README.md#int8-inference-instructions) [FP32](recommendation/tensorflow/wide_deep_large_ds/README.md#fp32-inference-instructions) |
| Recommendation         | TensorFlow    | [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf)         | Inference | [FP32](recommendation/tensorflow/wide_deep/README.md#fp32-inference-instructions) |

*Means the model belongs to [MLPerf](https://mlperf.org/) models and will be supported long-term.

**Means the BFloat16 data type support is experimental.
