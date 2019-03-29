# Image Recognition Model Optimization and Quantization with ResNet50

Content:
* [Goal](#goal)
* [Prerequisites](#prerequisites)
* [Install and Build TensorFlow Tools](#install-and-build-tensorflow-tools)
* [Floating point 32-bits Model Optimization](#fp32-model-optimization)
* [Floating point 32-bits Model Quantization to 8-bits Precision](#fp32-model-quantization-to-int8-precision)
* [Performance Evaluation](#performance-evaluation)

## Goal
Post-training model quantization and optimization objective is to:
* Reduce the model size, 
* Run faster inference (less latency), 
* Maintain the model performance (throughput and accuracy).

This is highly recommended in the case of mobile applications and systems of constrained memory and processing power.
Usually, there will be some loss in performance, but it has to be within the [acceptable range](#performance-evaluation). 

More resources: [Post-training quantization for mobile and IOT](https://www.tensorflow.org/lite/performance/post_training_quantization), and
[TensorFlow graph transform tool user guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms).

## Prerequisites
* The ResNet50 model topology graph (the `model graph_def` as `.pb` or `.pbtxt` file)
and the `checkpoint files` are required to start this tutorial.

## Install and Build TensorFlow Tools

* Clone the TensorFlow tools repository, and follow the [instructions](https://github.com/IntelAI/tools/tree/master/tensorflow-quantization)
for how to build the TensorFlow tools using Docker.
```
$ git clone https://github.com/IntelAI/tools.git
```

## FP32 Model Optimization
In this section, we assume that a trained model topology graph (the model graph_def as .pb or .pbtxt file) and the checkpoint files are available.
 * The `model graph_def` is used in `step 1` to get the possible **input and output node names** of the graph.
 * Both of the `model graph_def` and the `checkpoint file` are required in `step 2` to get the **model frozen graph**.
 * The `model frozen graph`, **optimized** (based on the graph structure and operations, etc.) in `step 3`.

We also assume that you are in the TensorFlow root directory (`/workspace/tensorflow/` inside the docker container) to execute the following steps.

1. Find out the possible input and output node names of the graph
    From the TensorFlow/tools root directory, run:
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
        --in_graph=/workspace/tensorflow/resnet50.pbtxt \
        --print_structure=false >& model_nodes.txt
    ```

    In the `model_nodes.txt` file, look for the input and output nodes names such as:
    ```
        Found 1 possible inputs: (name=input, type=float(1), shape=[?,224,224,3])
        Found 1 possible outputs: (name=predict, op=Softmax)
    ```
2. Freeze the graph where the checkpoint values are converted into constants in the graph:
    * The `--input_graph` is the model topology graph_def, and the checkpoint file are required.
    * The `--output_node_names` are obtained from step 1.
      >Note: `--input_graph` can be in either binary `pb` or text `pbtxt` format,
    and the `--input_binary` flag will be enabled or disabled accordingly.
    ```
        $ python tensorflow/python/tools/freeze_graph.py \
        --output_graph= /workspace/tensorflow/resnet50_frozen_fp32_graph.pb \
        --input_binary=False \
        --output_binary=True \
        --input_checkpoint=/workspace/tensorflow/resnet50_model.ckpt \
        --in_graph=/workspace/tensorflow/resnet50.pbtxt \
        --output_node_names=‘predict’
    ```
3. Optimize the FP32 frozen graph:
    * Set the `--in_graph` to the path of the model frozen graph (from step 2),
    * The `--inputs` and `--outputs` are the graph input and output node names (from step 1).
    * `--transforms` to be set based on the model graph structure (to remove unused nodes, combine operations, etc).
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=/workspace/tensorflow/freezed_resnet50.pb \
        --out_graph=/workspace/tensorflow/optimized_resnet50_fp32_graph.pb \
        --inputs='input'; \
        --outputs='predict'; \
        --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics)
        fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'
    ```
4. [Evaluate the model performance](#accuracy-for-fp32-optimized-graph) using the the optimized graph `optimized_resnet50_fp32_graph.pb` and check the model accuracy.

## FP32 Model Quantization to Int8 Precision
In this section, our objective is to quantize the output [FP32 optimized graph](#fp32-model-optimization) of the previous section
to `Int8` precision.
In case you did not do the FP32 model optimization by yourself, please follow the [instructions](/benchmarks/image_recognition/tensorflow/resnet50/README.md#fp32-inference-instructions) to download the Intel optimized
ResNet50 pre-trained model graph.

The following steps show how to convert the `FP32` model to `Int8` precision to reduce the model size:

5. Convert the FP32-graph to a dynamic range Int8-graph using the output node names (from step 1)

    ```
        $ python tensorflow/tools/quantization/quantize_graph.py \
        --input=/workspace/tensorflow/optimized_resnet50_fp32_graph.pb \
        --output=/workspace/tensorflow/int8_dynamic_range_resnet50_graph.pb \
        --output_node_names='predict' \
        --mode=eightbit \
        --intel_cpu_eightbitize=True
    ```

    [Evaluate the output int8 graph performance](#accuracy-for-int8-optimized-graph)
    to check the loss in performance after the model quantization.
    
    The log snippet for the dynamic range Int8 model accuracy:
    ```
        ...
        Processed 5100 images. (Top1 accuracy, Top5 accuracy) = (0.6665, 0.8506)
        Processed 5200 images. (Top1 accuracy, Top5 accuracy) = (0.6683, 0.8523)
        Processed 5300 images. (Top1 accuracy, Top5 accuracy) = (0.6698, 0.8538)
        ...
    ```

6. Convert from dynamic to static re-quantization range.
The following steps are to freeze the re-quantization range (also known as
calibration):

    In order to facilitate this section for the user, we attached a sample of the [`resnet50_min_max_log.txt` file](/docs/image_recognition/quantization/resnet50_min_max_log.txt).
    In case you decided to use it then you can skip the first two steps `Insert the logging op` and `Generate calibration data`. 

    * Insert the logging op:
        ```
            $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=/workspace/quantization/int8_resnet50_graph.pb \
            --out_graph=/workspace/quantization/logged_int8_resnet50.pb \
            --transforms='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'
        ```

    * Generate calibration data:
        * **Generate a data subset of the ImageNet dataset for calibration**, follow [instructions](/benchmarks/image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions)
          and run inference for accuracy (using `--accuracy-only`, `--in-graph=/home/<user>/optimized_resnet50_fp32_graph.pb` (from step 3),
            `--docker-image=intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl` and `-- calibration_only=True`).
          
          > Note: 
          > - `-- calibration_only=True` is a custom argument to be added at the end of the inference command as formatted (with a white space after `--`).
          > - This step works only with `--docker-image=intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl`, or an image generated using [TensorFlow](https://github.com/tensorflow/tensorflow) commit `7878f58d38915ba895670d3a550571bebd8c787c` or older.
          
          We run inference while generating calibration data to be able to pick images that are correctly classified with high confidence for calibration.
          The `optimized_resnet50_fp32_graph.pb` is used as the ResNet50 trained model at this step.
          A snippet of the ResNet50 inference results while generating the calibration data:
          ```
            Processed 10 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 20 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 30 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 40 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 50 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 60 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 70 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 80 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 90 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
            Processed 100 images. (Top1 accuracy, Top5 accuracy) = (1.0000, 1.0000)
          ```
          The calibration data `calibration-1-of-1` will be created in the current directory.
          ```
            $ mkdir dataset && cp calibration-1-of-1 dataset

          ```

        * **Generate the `resnet50_min_max_log.txt` file**, follow [instructions](/benchmarks/image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions)
            to run inference (using `--batch_size=10`, `--data-location=/home/<user>/dataset`, `--in-graph=/home/<user>/logged_int8_resnet50.pb`,
            `--accuracy-only`, and `-- calibrate=True`), and **store the output log in `resnet50_min_max_log.txt` file**.
            
            >Note:  
            `-- calibrate=True` is a custom argument to be added at the end of the inference command as formatted (with a white space after `--`).

        * The `resnet50_min_max_log.txt` file is used in the following step. We suggest that you store the `resnet50_min_max_log.txt` in the same location specified in
          the [start quantization process](https://github.com/IntelAI/tools/tree/master/tensorflow-quantization) section,
          which will be mounted inside the container to `/workspace/quantization`.
    
    * Run the calibration data replace the
        `RequantizationRangeOp` with constants in the original quantized graph (the output of step 1):
        ```
            $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=/workspace/quantization/int8_dynamic_range_resnet50_graph.pb \
            --out_graph=/workspace/quantization/freezed_range_int8_resnet50.pb \
            --transforms='freeze_requantization_ranges(min_max_log_file="/workspace/quantization/resnet50_min_max_log.txt")'
        ```
        
         [Evaluate the output int8 graph performance](#accuracy-for-int8-optimized-graph)
    to check the loss in performance after this step.
        A snippet of the inference log for accuracy:
        ```
            ...
            Processed 5100 images. (Top1 accuracy, Top5 accuracy) = (0.6529, 0.8647)
            Processed 5200 images. (Top1 accuracy, Top5 accuracy) = (0.6540, 0.8654)
            Processed 5300 images. (Top1 accuracy, Top5 accuracy) = (0.6555, 0.8664)
            ...
        ```

7. Fuse `RequantizeOp` with fused quantized convolutions, and generate the final
optimized Int8 graph
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=/workspace/quantization/freezed_range_int8_resnet50.pb \
        --out_graph=/workspace/quantization/final_int8_resnet50.pb \
        --outputs='predict' \
        --transforms='fuse_quantized_conv_and_requantize strip_unused_nodes'
    ```
    Check the final quantized ResNet50 model `final_int8_resnet50.pb` performance in
    the [Accuracy for Int8 Optimized Graph](#accuracy-for-int8-optimized-graph) section.


## Performance Evaluation

Validating the model performance is required after each step to verify if the output graph achieves the accuracy target.
* The model accuracy is used as a performance measure.
* The accuracy target is the optimized FP32 model accuracy values.
* The quantized `Int8` graph accuracy should not drop more than ~0.5-1%.


This section explains how to run ResNet50 inference and calculate the model accuracy using [Intel Model Zoo Benchmarks](https://github.com/IntelAI/models).

Clone the [IntelAI/models](https://github.com/IntelAI/models) repository, 
and follow the [documented steps](/benchmarks/image_recognition/tensorflow/resnet50/README.md#int8-inference-instructions)
to benchmark `ResNet50` inference performance for both FP32 and Int8 cases.

**Note that the benchmarking script should be run outside of the quantization docker container
and that some inputs to the benchmarking script are slightly different for `FP32` and `Int8` models (i.e. `--precision` and `--docker-image`).**


### Accuracy for FP32 Optimized Graph
Clone the [IntelAI/models](https://github.com/IntelAI/models) repository and follow the steps to run the FP32 benchmark
script to calculate `accuracy` and use the optimized FP32 graph in `--in-graph`.
   ```
        $ git clone https://github.com/IntelAI/models.git
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/<pretrained_model_directory>/optimized_resnet50_fp32_graph.pb \
            --model-name resnet50 \
            --framework tensorflow \
            --precision fp32 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/<dataset_directory> \
            --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
  ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
        ...
        Processed 4800 images. (Top1 accuracy, Top5 accuracy) = (0.7533, 0.9225)
        Processed 4900 images. (Top1 accuracy, Top5 accuracy) = (0.7531, 0.9227)
        Processed 5000 images. (Top1 accuracy, Top5 accuracy) = (0.7550, 0.9230)
        Processed 5100 images. (Top1 accuracy, Top5 accuracy) = (0.7545, 0.9224)
        Processed 5200 images. (Top1 accuracy, Top5 accuracy) = (0.7544, 0.9215)
        ...
   ```

### Accuracy for Int8 Optimized Graph

Clone the [IntelAI/models](https://github.com/IntelAI/models) repository and follow the steps to run the Int8 benchmark
script to calculate `accuracy` and use the Int8 graph in `--in-graph`.
   ```
        $ git clone https://github.com/IntelAI/models.git
        $ cd /home/<user>/models/benchmarks
        
        $ python launch_benchmark.py \
            --in-graph /home/<user>/<pretrained_model_directory>/final_resnet50_Int8_graph.pb \
            --model-name resnet50 \
            --framework tensorflow \
            --precision int8 \
            --mode inference \
            --accuracy-only \
            --batch-size=100 \
            --socket-id 0 \
            --data-location /home/<user>/<dataset_directory> \
            --docker-image intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl
   ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
        ...
        Processed 4500 images. (Top1 accuracy, Top5 accuracy) = (0.7384, 0.9207)
        Processed 4600 images. (Top1 accuracy, Top5 accuracy) = (0.7387, 0.9209)
        Processed 4700 images. (Top1 accuracy, Top5 accuracy) = (0.7383, 0.9211)
        Processed 4800 images. (Top1 accuracy, Top5 accuracy) = (0.7375, 0.9208)
        Processed 4900 images. (Top1 accuracy, Top5 accuracy) = (0.7382, 0.9212)
        Processed 5000 images. (Top1 accuracy, Top5 accuracy) = (0.7378, 0.9210)
        Processed 5100 images. (Top1 accuracy, Top5 accuracy) = (0.7380, 0.9214)
        Processed 5200 images. (Top1 accuracy, Top5 accuracy) = (0.7387, 0.9219)
        Processed 5300 images. (Top1 accuracy, Top5 accuracy) = (0.7387, 0.9221)
        Processed 5400 images. (Top1 accuracy, Top5 accuracy) = (0.7376, 0.9213)
        Processed 5500 images. (Top1 accuracy, Top5 accuracy) = (0.7373, 0.9211)
        ...
   ```