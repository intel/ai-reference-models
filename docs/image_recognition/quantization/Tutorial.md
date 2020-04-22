# Image Recognition Model Optimization and Quantization with ResNet50 and ResNet50v1.5

Content:
* [Goal](#goal)
* [Prerequisites](#prerequisites)
* [Floating point 32-bits Model Quantization to 8-bits Precision](#floating-point-32-bits-model-quantization-to-8-bits-precision)
* [Performance Evaluation](#performance-evaluation)

## Goal
Post-training model quantization and optimization objective is to:
* Reduce the model size, 
* Run faster online inference (batch size = 1), 
* Maintain the model performance (larger batch inference and accuracy).

This is highly recommended in the case of mobile applications and systems of constrained memory and processing power.
Usually, there will be some loss in performance, but it has to be within the [acceptable range](#performance-evaluation). 

More resources: [Post-training quantization for mobile and IOT](https://www.tensorflow.org/lite/performance/post_training_quantization), and
[TensorFlow graph transform tool user guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms).

This tutorial provides a step-by-step guide for ResNet50 and ResNet50v1.5 models conversion from Floating Point 32-bits (FP32) precision to 8-bits Precision (INT8) using [Intel® AI Quantization Tools for TensorFlow](https://github.com/IntelAI/tools).
## Prerequisites
* The binary installed [Intel® optimizations for TensorFlow 2.1.0](https://pypi.org/project/intel-tensorflow/).
```
    $ pip install intel-tensorflow==2.1.0
    $ pip install intel-quantization
```

* The source release repository of [Model Zoo](https://github.com/IntelAI/models) for Intel® Architecture.
```
    $ cd ~
    $ git clone https://github.com/IntelAI/models.git
```
* The source release repository of [Intel® AI Quantization Tools for TensorFlow](https://github.com/IntelAI/tools).
```
    $ cd ~
    $ git clone https://github.com/IntelAI/tools.git
```

* The frozen FP32 pre-trained model and the ImageNet dataset will be required for fully automatic quantization.
The TensorFlow models repository provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process, and convert the ImageNet dataset to the TFRecord format.


## Floating point 32-bits Model Quantization to 8-bits Precision

In this section, we assume that the ImageNet dataset is available, and also you can download the FP32 pre-trained model as shown in [ResNet50](#resnet50) and [ResNet50v1.5](#resnet50v1.5) instructions.

[Intel® AI Quantization Tools for TensorFlow](https://github.com/IntelAI/tools) repository provides a python script which fully automates the quantization steps for ResNet50 and ResNet50v1.5.
The quantization script requires the input parameters of pre-trained model, dataset path to match with your local environment.
And then simply specify the model name and execute the python script, you will get the fully automatic quantization conversion from FP32 to INT8.
```
    $ cd /home/<user>/tools
    $ python api/examples/quantize_model_zoo.py \
        --model model \
        --in_graph /home/<user>/fp32_pretrained_model.pb \
        --out_graph /home/<user>/output.pb \
        --data_location /home/<user>/dataset \
        --models_zoo_location /home/<user>/models
```

The `quantize_model_zoo.py` script executes the following steps to optimize and quantize a FP32 model:
1) Optimize fp32_frozen_graph based on the graph structure and operations, etc.
2) Quantize graph: The FP32-graph is converted to a dynamic range INT8 graph using the output node names.
3) Calibration: It converts the dynamic re-quantization range (`RequantizationRangeOp`) in the initially quantized graph to static (constants).
4) Fuse `RequantizeOp` with fused quantized convolutions, and generate the final
optimized INT8 graph.

For tuning the pre-defined graph quantization parameters such as
(`INPUT_NODE_LIST`, `OUTPUT_NODE_LIST`, `EXCLUDED_OPS_LIST`, `EXCLUDED_NODE_LIST`, enable or disable `PER_CHANNEL_FLAG`), please check the [models.json](https://github.com/IntelAI/tools/blob/master/api/config/models.json) file, and the [quantization API documentation](https://github.com/IntelAI/tools/tree/master/api#integration-with-model-zoo-for-intel-architecture).


## ResNet50

* Download the FP32 ResNet50 pre-trained model to a location of your choice or as suggested:
```
    $ cd /home/<user>/tools/api/models/resnet50
    $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb
```
* Run the automatic quantization script with the input parameters of pre-trained model, dataset path to match with your local environment.
And then, you will get the quantized ResNet50 INT8 pre-trained model saved in `/home/<user>/tools/api/models/resnet50/resnet50_int8.pb` as specified.
```
    $ cd /home/<user>/tools
    $ python api/examples/quantize_model_zoo.py \
        --model resnet50 \
        --in_graph /home/<user>/tools/api/models/resnet50/resnet50_fp32_pretrained_model.pb \
        --out_graph /home/<user>/tools/api/models/resnet50/resnet50_int8.pb \
        --data_location /home/<user>/imagenet \
        --models_zoo_location /home/<user>/models
```

* An example for the log output when the graph quantization run completes:
```
    Model Config: MODEL_NAME:resnet50
    Model Config: LAUNCH_BENCHMARK_PARAMS:{'LAUNCH_BENCHMARK_SCRIPT': 'benchmarks/launch_benchmark.py', 'LAUNCH_BENCHMARK_CMD': ['--model-name resnet50', '--framework tensorflow', '--precision int8', '--mode inference', '--batch-size 100', '--accuracy-only'], 'IN_GRAPH': '--in-graph {}', 'DATA_LOCATION': '--data-location {}'}
    Model Config: QUANTIZE_GRAPH_CONVERTER_PARAMS:{'INPUT_NODE_LIST': ['input'], 'OUTPUT_NODE_LIST': ['predict'], 'EXCLUDED_OPS_LIST': [], 'EXCLUDED_NODE_LIST': [], 'PER_CHANNEL_FLAG': False}
    Model Config: Supported models - ['resnet50', 'resnet50v1_5', 'resnet101', 'mobilenet_v1', 'ssd_mobilenet', 'ssd_resnet34', 'faster_rcnn', 'rfcn', 'inceptionv3']
    Inference Calibration Command: python /home/<user>/models/benchmarks/launch_benchmark.py --model-name resnet50 --framework tensorflow --precision int8 --mode inference --batch-size 100 --accuracy-only --data-location /home/<user>/imagenet --in-graph {}
    ...
    
    ;v0/resnet_v115/conv51/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][2.67806506]
    ;v0/resnet_v115/conv52/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][23.9200363]
    ;v0/mpool0/MaxPool_eightbit_max_v0/conv0/Relu__print__;__max:[5.72005272];v0/mpool0/MaxPool_eightbit_min_v0/conv0/Relu__print__;__min:[-0]
    ...
    
    Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7386, 0.9168)
    Iteration time: 1.3564 ms
    Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7387, 0.9169)
    Iteration time: 1.3461 ms
    Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7387, 0.9169)
    Ran inference with batch size 100
    Log location outside container: /home/<user>/models/benchmarks/common/tensorflow/logs/benchmark_resnet50_inference_int8_20200401_115400.log
    I0401 12:05:21.515716 139714463500096 graph_converter.py:195] Converted graph file is saved to: /home/<user>/output.pb
```

## ResNet50v1.5

* Download the FP32 ResNet50v1.5 pre-trained model to a location of your choice or as suggested:
```
    $ mkdir /home/<user>/tools/api/models/resnet50v1_5 && cd /home/<user>/tools/api/models/resnet50v1_5
    $ wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
```
* Run the automatic quantization script with the input parameters of pre-trained model, dataset path to match with your local environment.
And then, you will get the quantized ResNet50v1.5 INT8 pre-trained model saved in `/home/<user>/tools/api/models/resnet50v1_5/resnet50v1_5_int8.pb` as specified.
```
    $ cd /home/<user>/tools
    $ python api/examples/quantize_model_zoo.py \
        --model resnet50v1_5 \
        --in_graph /home/<user>/tools/api/models/resnet50v1_5/resnet50_v1.pb \
        --out_graph /home/<user>/tools/api/models/resnet50v1_5/resnet50v1_5_int8.pb \
        --data_location /home/<user>/imagenet \
        --models_zoo_location /home/<user>/models
```

* An example for the log output when the graph quantization run completes:
```
Model Config: MODEL_NAME:resnet50v1_5
Model Config: LAUNCH_BENCHMARK_PARAMS:{'LAUNCH_BENCHMARK_SCRIPT': 'benchmarks/launch_benchmark.py', 'LAUNCH_BENCHMARK_CMD': ['--model-name resnet50v1_5', '--framework tensorflow', '--precision int8', '--mode inference', '--batch-size 100', '--accuracy-only'], 'IN_GRAPH': '--in-graph {}', 'DATA_LOCATION': '--data-location {}'}
Model Config: QUANTIZE_GRAPH_CONVERTER_PARAMS:{'INPUT_NODE_LIST': ['input_tensor'], 'OUTPUT_NODE_LIST': ['ArgMax', 'softmax_tensor'], 'EXCLUDED_OPS_LIST': [], 'EXCLUDED_NODE_LIST': [], 'PER_CHANNEL_FLAG': True}
Model Config: Supported models - ['resnet50', 'resnet50v1_5', 'resnet101', 'mobilenet_v1', 'ssd_mobilenet', 'ssd_resnet34', 'faster_rcnn', 'rfcn', 'inceptionv3']
Inference Calibration Command: python /home/<user>/models/benchmarks/launch_benchmark.py --model-name resnet50v1_5 --framework tensorflow --precision int8 --mode inference --batch-size 100 --accuracy-only --data-location /home/<user>/imagenet --in-graph {}
...

;resnet_model/conv2d_5/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][16.3215694]
;resnet_model/conv2d_6/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][13.4745159]
;resnet_model/conv2d_7/Conv2D_eightbit_requant_range__print__;__requant_min_max:[0][14.5196199]
...

Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7628, 0.9299)
Iteration time: 1.8439 ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7627, 0.9298)
Iteration time: 1.8366 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7628, 0.9298)
Ran inference with batch size 100
Log location outside container: /home/<user>/models/benchmarks/common/tensorflow/logs/benchmark_resnet50v1_5_inference_int8_20200402_125005.log
I0402 13:07:13.125293 140357697517376 graph_converter.py:195] Converted graph file is saved to: api/models/resnet50v1_5/resnet50v1_5_int8.pb
```

## Performance Evaluation

Verify the quantized model performance:

* Run inference using the final quantized graph and calculate the accuracy.
* Typically, the accuracy target is the optimized FP32 model accuracy values.
* The quantized INT8 graph accuracy should not drop more than ~0.5-1%.

### ResNet50 Accuracy Evaluation:
Check [IntelAI/models](https://github.com/IntelAI/models) repository and [ResNet50 README](/benchmarks/image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions)
for TensorFlow models inference benchmarks with different precisions.

#### FP32
Follow the steps in [ResNet50 README](/benchmarks/image_recognition/tensorflow/resnet50/README.md#fp32-inference-instructions) to run the FP32 
script to calculate `accuracy` and use the FP32 graph in `--in-graph`.
   ```
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/tools/api/models/resnet50/resnet50_fp32_pretrained_model.pb \
            --model-name resnet50 \
            --framework tensorflow \
            --precision fp32 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/imagenet \
            --docker-image intel/intel-optimized-tensorflow:2.1.0
  ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
        ...
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7422, 0.9184)
        Iteration time: 0.3590 ms
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7423, 0.9184)
        Iteration time: 0.3608 ms
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7424, 0.9184)
        Iteration time: 0.3555 ms
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7425, 0.9185)
        Iteration time: 0.3561 ms
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7425, 0.9185)
        ...
   ```

#### INT8

Follow the steps in [ResNet50 README](/benchmarks/image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions)
to run the INT8 script to calculate `accuracy` and use the path to the `resnet50_int8.pb` INT8 graph in `--in-graph`.
   ```
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/tools/api/models/resnet50/resnet50_int8.pb \
            --model-name resnet50 \
            --framework tensorflow \
            --precision int8 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/imagenet \
            --docker-image intel/intel-optimized-tensorflow:2.1.0
   ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
        ...
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7369, 0.9159)
        Iteration time: 0.1961 ms
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7370, 0.9160)
        Iteration time: 0.1967 ms
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7371, 0.9159)
        Iteration time: 0.1952 ms
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7371, 0.9160)
        Iteration time: 0.1968 ms
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7371, 0.9160)
        ...
   ```


### ResNet50v1.5 Accuracy Evaluation:
Check [IntelAI/models](https://github.com/IntelAI/models) repository and [ResNet50v1.5 README](/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#int8-inference-instructions)
for TensorFlow models inference benchmarks with different precisions.

#### FP32
Follow the steps in [ResNet50v1.5 README](/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#fp32-inference-instructions) to run the FP32 
script to calculate `accuracy` and use the FP32 graph in `--in-graph`.
   ```
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/tools/api/models/resnet50v1_5/resnet50_v1.pb \
            --model-name resnet50v1_5 \
            --framework tensorflow \
            --precision fp32 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/imagenet \
            --docker-image intel/intel-optimized-tensorflow:2.1.0
  ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
        ...
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7647, 0.9306)
        Iteration time: 0.4688 ms
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7647, 0.9306)
        Iteration time: 0.4694 ms
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7648, 0.9307)
        Iteration time: 0.4664 ms
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7647, 0.9307)
        Iteration time: 0.4650 ms
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7648, 0.9308)
        ...
   ```

#### INT8

Follow the steps in [ResNet50v1.5 README](/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#int8-inference-instructions)
to run the INT8 script to calculate `accuracy` and use the path to the `resnet50v1_5_int8.pb` INT8 graph in `--in-graph`.
   ```
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/tools/api/models/resnet50v1_5/resnet50v1_5_int8.pb \
            --model-name resnet50v1_5 \
            --framework tensorflow \
            --precision int8 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/imagenet \
            --docker-image intel/intel-optimized-tensorflow:2.1.0
   ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
        ...
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7614, 0.9298)
        Iteration time: 0.2126 ms
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7614, 0.9298)
        Iteration time: 0.2125 ms
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7614, 0.9298)
        Iteration time: 0.2128 ms
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7614, 0.9298)
        Iteration time: 0.2122 ms
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7616, 0.9298)
       ...
   ```


##
Check [Intel® AI Quantization Tools for TensorFlow](https://github.com/IntelAI/tools/tree/master/api#quantization-python-programming-api-quick-start)
for more details about the quantization scripts, procedures with different models. And for [Docker support](https://github.com/IntelAI/tools/tree/master/api#docker-support).

