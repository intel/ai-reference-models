# Using Intel® Model Zoo on Windows Systems

## Prerequisites for running on bare metal

Basic requirements for running all TensorFlow models on Windows include:
 * Install [Python 3.7+ 64-bit release for Windows](https://www.python.org/downloads/windows/), and add it to your system `%PATH%` environment variable.
 * Download and Install [Microsoft Visual C++ 2022 Redistributable](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).
 * [MSYS2](https://www.msys2.org). If MSYS2 is installed to `C:\msys64`, add `C:\msys64\usr\bin` to your `%PATH%` environment variable.
 Add `MSYS64_BASH=C:\msys64\usr\bin\bash.exe` environment variable to your system.
 Then, using `cmd.exe`, run:
 ```
    pacman -S git patch unzip
 ```
 * Install [intel-tensorflow](https://pypi.org/project/intel-tensorflow/)
 * Install the common models dependencies:
     * python-tk
     * libsm6
     * libxext6
     * requests
 
Individual models may have additional dependencies that need to be
installed before running it. Please follow the instructions in each model documentation. 

## Run TensorFlow models
The following list of models are tested on Windows, please check each model instructions from the `Run from the Model Zoo repository` column based on the available precisions.
>Note that on Windows systems all of the system cores will be used. 
>For users of Windows desktop/laptops, it is strongly encouraged to instead use the batch file provided [here](/benchmarks/common/windows_intel1dnn_setenv.bat) to open a Windows command prompt pre-configured with optimized settings to achieve high AI workload performance on Intel hardware (e.g. TigerLake & AlderLake) for image recognition models.

| Use Case                | Model              | Mode      | Run from the Model Zoo repository |
| ----------------------- | ------------------ | --------- | --------------------------------- |
| Image Recognition       | [DenseNet169](https://arxiv.org/pdf/1608.06993.pdf) | Inference | [FP32](/benchmarks/image_recognition/tensorflow/densenet169/inference/fp32/README.md) |
| Image Recognition       | [Inception V3](https://arxiv.org/pdf/1512.00567.pdf) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/inceptionv3/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/inceptionv3/inference/fp32/README.md) |
| Image Recognition       | [Inception V4](https://arxiv.org/pdf/1602.07261.pdf) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/inceptionv4/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/inceptionv4/inference/fp32/README.md) |
| Image Recognition       | [MobileNet V1*](https://arxiv.org/pdf/1704.04861.pdf) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/mobilenet_v1/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/mobilenet_v1/inference/fp32/README.md) |
| Image Recognition       | [ResNet 101](https://arxiv.org/pdf/1512.03385.pdf) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/resnet101/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet101/inference/fp32/README.md) |
| Image Recognition       | [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf) | Inference  | [Int8](/benchmarks/image_recognition/tensorflow/resnet50/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet50/inference/fp32/README.md) |
| Image Recognition       | [ResNet 50v1.5](https://github.com/tensorflow/models/tree/master/official/resnet) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/fp32/README.md) |
| Image Segmentation      | [3D U-Net MLPerf](https://arxiv.org/pdf/1606.06650.pdf) | Inference | [FP32](/benchmarks/image_segmentation/tensorflow/3d_unet_mlperf/README.md) |
| Language Modeling       | [BERT](https://arxiv.org/pdf/1810.04805.pdf) | Inference | [FP32](/benchmarks/language_modeling/tensorflow/bert_large/inference/fp32/README.md) |
| Language Translation    | [BERT](https://arxiv.org/pdf/1810.04805.pdf) | Inference | [FP32](/benchmarks/language_translation/tensorflow/bert/README.md#fp32-inference-instructions) |
| Language Translation    | [Transformer_LT_Official](https://arxiv.org/pdf/1706.03762.pdf) | Inference | [FP32](/benchmarks/language_translation/tensorflow/transformer_lt_official/inference/fp32/README.md) |
| Object Detection        | [R-FCN](https://arxiv.org/pdf/1605.06409.pdf) | Inference | [Int8](/benchmarks/object_detection/tensorflow/rfcn/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/rfcn/inference/fp32/README.md) |
| Object Detection        | [SSD-MobileNet*](https://arxiv.org/pdf/1704.04861.pdf) | Inference | [Int8](/benchmarks/object_detection/tensorflow/ssd-mobilenet/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/ssd-mobilenet/inference/fp32/README.md) |
| Object Detection        | [SSD-ResNet34*](https://arxiv.org/pdf/1512.02325.pdf) | Inference | [Int8](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/fp32/README.md) |
| Recommendation          | [DIEN](https://arxiv.org/abs/1809.03672) | Inference | [FP32](/benchmarks/recommendation/tensorflow/dien#fp32-inference) |
| Recommendation          | [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf) | Inference | [FP32](/benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/README.md) |
