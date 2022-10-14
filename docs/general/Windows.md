# Using Intel® Model Zoo on Windows Systems

## Prerequisites for running on bare metal

Basic requirements for running all TensorFlow models on Windows include:
 * Install [Python 3.7+ 64-bit release for Windows](https://www.python.org/downloads/windows/), and add it to your system `%PATH%` environment variable.
 * Download and Install [Microsoft Visual C++ 2022 Redistributable](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).
 * [MSYS2](https://www.msys2.org). If MSYS2 is installed to `C:\msys64`, add `C:\msys64\usr\bin` to your `%PATH%` environment variable.
 Then, using `cmd.exe`, run:
 ```
    pacman -S git patch unzip
 ```

## TensorFlow models
* Install [intel-tensorflow](https://pypi.org/project/intel-tensorflow/)
* Set `MSYS64_BASH=C:\msys64\usr\bin\bash.exe` environment variable to your system. The path may change based on where have you installed MSYS2 on our system.
* Install the common models dependencies:
     * python-tk
     * libsm6
     * libxext6
     * requests
 
Individual models may have additional dependencies that need to be
installed before running it. Please follow the instructions in each model documentation. 

The following list of models are tested on Windows, please check each model instructions from the `Model Documentation` column based on the available precisions.
>Note that on Windows systems all of the system cores will be used. 
>For users of Windows desktop/laptops, it is strongly encouraged to instead use the batch file provided [here](/benchmarks/common/windows_intel1dnn_setenv.bat) to open a Windows command prompt pre-configured with optimized settings to achieve high AI workload performance on Intel hardware (e.g. TigerLake & AlderLake) for image recognition models.

| Use Case                | Model              | Mode      | Model Documentation |
| ----------------------- | ------------------ | --------- | --------------------------------- |
| Image Recognition       | [DenseNet169](https://arxiv.org/pdf/1608.06993.pdf) | Inference | [FP32](/benchmarks/image_recognition/tensorflow/densenet169/inference/fp32/README.md) |
| Image Recognition       | [Inception V3](https://arxiv.org/pdf/1512.00567.pdf) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/inceptionv3/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/inceptionv3/inference/fp32/README.md) |
| Image Recognition       | [Inception V4](https://arxiv.org/pdf/1602.07261.pdf) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/inceptionv4/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/inceptionv4/inference/fp32/README.md) |
| Image Recognition       | [MobileNet V1*](https://arxiv.org/pdf/1704.04861.pdf) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/mobilenet_v1/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/mobilenet_v1/inference/fp32/README.md) |
| Image Recognition       | [ResNet 101](https://arxiv.org/pdf/1512.03385.pdf) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/resnet101/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet101/inference/fp32/README.md) |
| Image Recognition       | [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf) | Inference  | [Int8](/benchmarks/image_recognition/tensorflow/resnet50/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet50/inference/fp32/README.md) |
| Image Recognition       | [ResNet 50v1.5](https://github.com/tensorflow/models/tree/master/official/resnet) | Inference | [Int8](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/int8/README.md) [FP32](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/fp32/README.md) |
| Image Segmentation      | [3D U-Net MLPerf](https://arxiv.org/pdf/1606.06650.pdf) | Inference | [FP32 BFloat16](/benchmarks/image_segmentation/tensorflow/3d_unet_mlperf/inference/README.md) |
| Language Modeling       | [BERT](https://arxiv.org/pdf/1810.04805.pdf) | Inference | [FP32](/benchmarks/language_modeling/tensorflow/bert_large/inference/fp32/README.md) |
| Language Translation    | [BERT](https://arxiv.org/pdf/1810.04805.pdf) | Inference | [FP32](/benchmarks/language_translation/tensorflow/bert/README.md#fp32-inference-instructions) |
| Language Translation    | [Transformer_LT_Official](https://arxiv.org/pdf/1706.03762.pdf) | Inference | [FP32](/benchmarks/language_translation/tensorflow/transformer_lt_official/inference/fp32/README.md) |
| Object Detection        | [R-FCN](https://arxiv.org/pdf/1605.06409.pdf) | Inference | [Int8](/benchmarks/object_detection/tensorflow/rfcn/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/rfcn/inference/fp32/README.md) |
| Object Detection        | [SSD-MobileNet*](https://arxiv.org/pdf/1704.04861.pdf) | Inference | [Int8](/benchmarks/object_detection/tensorflow/ssd-mobilenet/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/ssd-mobilenet/inference/fp32/README.md) |
| Object Detection        | [SSD-ResNet34*](https://arxiv.org/pdf/1512.02325.pdf) | Inference | [Int8](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/int8/README.md) [FP32](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/fp32/README.md) |
| Recommendation          | [DIEN](https://arxiv.org/abs/1809.03672) | Inference | [FP32](/benchmarks/recommendation/tensorflow/dien#fp32-inference) |
| Recommendation          | [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf) | Inference | [FP32](/benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/README.md) |


## PyTorch models
Install [PyTorch](https://pytorch.org/)
```
pip install torch torchvision
```

The following list of models are tested on Windows, please check each model instructions from the `Model Documentation` column based on the available precisions.

| Use Case                | Model              | Mode      | Model Documentation |
| ----------------------- | ------------------ | --------- | ------------------- |
| Image Recognition       | [GoogLeNet](https://arxiv.org/abs/1409.4842)         | Inference | [FP32](/quickstart/image_recognition/pytorch/googlenet/inference/cpu/README.md) |
| Image Recognition       | [Inception v3](https://arxiv.org/pdf/1512.00567.pdf) | Inference | [FP32](/quickstart/image_recognition/pytorch/inception_v3/inference/cpu/README.md) |
| Image Recognition       | [MNASNet 0.5](https://arxiv.org/abs/1807.11626)      | Inference | [FP32](/quickstart/image_recognition/pytorch/mnasnet0_5/inference/cpu/README.md) |
| Image Recognition       | [MNASNet 1.0](https://arxiv.org/abs/1807.11626)      | Inference | [FP32](/quickstart/image_recognition/pytorch/mnasnet1_0/inference/cpu/README.md) |
| Image Recognition       | [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)    | Inference | [FP32](/quickstart/image_recognition/pytorch/resnet50/inference/cpu/fp32/README.md) [BFloat16](/quickstart/image_recognition/pytorch/resnet50/inference/cpu/bfloat16/README.md) |
| Image Recognition       | [ResNet 101](https://arxiv.org/pdf/1512.03385.pdf)   | Inference | [FP32](/quickstart/image_recognition/pytorch/resnet101/inference/cpu/README.md) |
| Image Recognition       | [ResNet 152](https://arxiv.org/pdf/1512.03385.pdf)   | Inference | [FP32](/quickstart/image_recognition/pytorch/resnet152/inference/cpu/README.md) |
| Image Recognition       | [ResNext 32x4d](https://arxiv.org/abs/1611.05431)    | Inference | [FP32](/quickstart/image_recognition/pytorch/resnext-32x4d/inference/cpu/README.md) |
| Image Recognition       | [ResNext 32x16d](https://arxiv.org/abs/1611.05431)   | Inference | [FP32](/quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/README.md) [BFloat16](/quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/README.md) |
| Image Recognition       | [VGG-11](https://arxiv.org/abs/1409.1556)            | Inference | [FP32](/quickstart/image_recognition/pytorch/vgg11/inference/cpu/README.md) |
| Image Recognition       | [VGG-11 with batch normalization](https://arxiv.org/abs/1409.1556) | Inference | [FP32](/quickstart/image_recognition/pytorch/vgg11_bn/inference/cpu/README.md) |
| Image Recognition       | [Wide ResNet-50-2](https://arxiv.org/pdf/1605.07146.pdf)   | Inference | [FP32](/quickstart/image_recognition/pytorch/wide_resnet50_2/inference/cpu/README.md) |
| Image Recognition       | [Wide ResNet-101-2](https://arxiv.org/pdf/1605.07146.pdf)  | Inference | [FP32](/quickstart/image_recognition/pytorch/wide_resnet101_2/inference/cpu/README.md) |
| Object Detection        | [Faster R-CNN ResNet50 FPN](https://arxiv.org/abs/1506.01497) | Inference  | [FP32](/quickstart/object_detection/pytorch/faster_rcnn_resnet50_fpn/inference/cpu/README.md) |
| Object Detection        | [Mask R-CNN](https://arxiv.org/abs/1703.06870)                | Inference  | [FP32](/quickstart/object_detection/pytorch/maskrcnn/inference/cpu/README.md) |
| Object Detection        | [Mask R-CNN ResNet50 FPN](https://arxiv.org/abs/1703.06870)   | Inference  | [FP32](/quickstart/object_detection/pytorch/maskrcnn_resnet50_fpn/inference/cpu/README.md) |
| Object Detection        | [RetinaNet ResNet-50 FPN](https://arxiv.org/abs/1708.02002)   | Inference  | [FP32](/quickstart/object_detection/pytorch/retinanet_resnet50_fpn/inference/cpu/README.md) |
| Shot Boundary Detection          | [TransNetV2](https://arxiv.org/pdf/2008.04838.pdf)         | Inference  | [FP32](/quickstart/shot_boundary_detection/pytorch/transnetv2/inference/cpu/README.md) |
