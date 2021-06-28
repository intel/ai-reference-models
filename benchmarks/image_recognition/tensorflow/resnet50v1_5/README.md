# ResNet50 (v1.5)

The following links have instructions for how to run ResNet50 (v1.5) for the
following modes/precisions:
* [Int8 inference](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/int8/README.md)
* [FP32 inference](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/fp32/README.md)
* [BFloat16 inference](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/bfloat16/README.md)
* [FP32 training](/benchmarks/image_recognition/tensorflow/resnet50v1_5/training/fp32/README.md)
* [BFloat16 training](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/bfloat16/README.md)

The original ResNet model has multiple versions which have shown better accuracy
and/or batch inference and training performance. As mentioned in TensorFlow's [official ResNet
model page](https://github.com/tensorflow/models/tree/master/official/resnet), 3 different
versions of the original ResNet model exists - ResNet50v1, ResNet50v1.5, and ResNet50v2.
As a side note, ResNet50v1.5 is also in MLPerf's [cloud inference benchmark for
image classification](https://github.com/mlperf/inference/tree/master/cloud/image_classification)
and [training benchmark](https://github.com/mlperf/training).
