#  Wide and Deep model Optimization and Quantization

Content:
* [Goal](#goal)
* [Prerequisites](#prerequisites)
* [Install and Build TensorFlow Tools](#install-and-build-tensorflow-tools)
* [Floating point 32-bits Model Optimization](#fp32-model-optimization)
* [Floating point 32-bits Model Quantization to 8-bits Precision](#fp32-model-quantization-to-int8-precision)
* [Performance Evaluation](#performance-evaluation)

## Goal
Post-training model quantization and optimization objective is to:
* Reduce the model size
* Run faster online inference

This is highly recommended in the case of mobile applications and systems of constrained memory and processing power.
Usually, there will be some loss in accuracy, but it has to be within the [acceptable range](#performance-evaluation).

More resources: [Post-training quantization for mobile and IOT](https://www.tensorflow.org/lite/performance/post_training_quantization), and
[TensorFlow graph transform tool user guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms).

## Prerequisites
* The wide and deep saved model graph generated during training


## Install and Build TensorFlow Tools

Build an image which contains transform_graph and summarize_graph tools. The
initial build may take a long time, but subsequent builds will be quicker
since layers are cached
 ```
     $ git clone https://github.com/IntelAI/tools.git
     cd tools/tensorflow_quantization

     docker build \
     --build-arg HTTP_PROXY=${HTTP_PROXY} \
     --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
     --build-arg http_proxy=${http_proxy} \
     --build-arg https_proxy=${https_proxy} \
     -t quantization:latest -f Dockerfile .
 ```
Launch quantization script launch_quantization.py by providing args as below,
this will get user into container environment (/workspace/tensorflow/) with
quantization tools.

 ```
      --docker-image: Docker image tag from above step (quantization:latest)
      --pre-trained-model-dir: Path to your pre-trained model directory, which will
      be mounted inside container at /workspace/quantization.

      python launch_quantization.py \
      --docker-image quantization:latest \
      --pre-trained-model-dir /home/<user>/<pre_trained_model_dir>
 ```
Please provide the output graphs locations relative to /workspace/quantization, so that results are written back to
local machine

## FP32 Model Optimization
In this section, we assume that a saved model graph generated during training is available.
 * The `model graph_def` is used in `step 1` to get the possible **input and output node names** of the graph.
 * The input saved model directory generated during training is used in `step 2` to get the **model frozen graph**.
 * The `model frozen graph`, **optimized** (based on the graph structure and operations, etc.) in `step 3`.

We also assume that you are in the TensorFlow root directory (`/workspace/tensorflow/` inside the docker container) to execute the following steps.

1. Find out the possible input and output node names of the graph
    From the TensorFlow/tools root directory, run:
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
        --in_graph=/workspace/quantization/graph.pbtxt \
        --print_structure=false >& model_nodes.txt
    ```

    In the `model_nodes.txt` file, look for the input and output nodes names such as:
    ```
        Found 1 possible inputs: (name=input, type=float(1), shape=[?,224,224,3])
        Found 1 possible outputs: (name=predict, op=Softmax)
    ```
2. Freeze the graph:
    * The `--input_saved_model_dir` is the topology saved model directory generated during training
    * The `--output_node_names` are obtained from step 1.
      >Note: `--input_graph` can be in either binary `pb` or text `pbtxt` format
    ```
        $ python tensorflow/python/tools/freeze_graph.py \
        --input_saved_model_dir=/workspace/tensorflow/model_<>/exports/<> \
        --output_graph= /workspace/quantization/wide_deep_frozen_fp32_graph.pb \
        --output_node_names=head/predictions/probabilities
    ```
3. Optimize the FP32 frozen graph to remove training and unused nodes:
    * Set the `--in_graph` to the path of the model frozen graph (from step 2),
    * The `--inputs` and `--outputs` are the graph input and output node names (from step 1).
    * `--transforms` to be set based on the model graph structure (to remove unused nodes, combine operations, etc).
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=/workspace/quantization/wide_deep_frozen_fp32_graph.pb \
        --out_graph=/workspace/quantization/wide_deep_fp32_graph.pb \
        --inputs='Placeholder,Placeholder_1,Placeholder_2,Placeholder_3,Placeholder_4, \
        Placeholder_5,Placeholder_6,Placeholder_7,Placeholder_8,Placeholder_9,Placeholder_10, \
        Placeholder_11,Placeholder_12,Placeholder_13,Placeholder_14,Placeholder_15,Placeholder_16, \
        Placeholder_17,Placeholder_18,Placeholder_19,Placeholder_20,Placeholder_21,Placeholder_22, \
        Placeholder_23,Placeholder_24,Placeholder_25,Placeholder_26,Placeholder_27,Placeholder_28,  \
        Placeholder_29,Placeholder_30,Placeholder_31,Placeholder_32,Placeholder_33,Placeholder_34,   \
        Placeholder_35,Placeholder_36,Placeholder_37,Placeholder_38' \
        --outputs='head/predictions/probabilities' \
        --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) remove_attribute(attribute_name=_class)'
    ```
4. Feature Column optimization of FP32 graph:

    Clone the [IntelAI/models](https://github.com/IntelAI/models) repository and use featurecolumn_graph_optimization.py script by setting `--input-graph` to the path of Fp32 graph obtained from above step and enable flag  `wide_and_deep_large_ds` to perform model specific optimizations(fusion of categorical and numeric columns to accept preprocessed and fused data).
    ```
            $ git clone https://github.com/IntelAI/models.git
            $ cd /home/<user>/models
            $ python models/recommendation/tensorflow/wide_deep_large_ds/dataset/featurecolumn_graph_optimization.py \
            --input-graph /workspace/quantization/wide_deep_fp32_graph.pb \
            --output-graph /workspace/quantization/optimized_wide_deep_fp32_graph.pb \
            --output-nodes head/predictions/probabilities \
            --wide_and_deep_large_ds True
      ```
5. [Evaluate the model performance](#accuracy-for-fp32-optimized-graph) using
the the optimized graph `optimized_wide_deep_fp32_graph.pb` and check the model accuracy.

## FP32 Model Quantization to Int8 Precision
In this section, our objective is to quantize the output [FP32 optimized graph](#fp32-model-optimization) of the previous section
to `Int8` precision.
In case you did not do the FP32 model optimization by yourself, please follow the [instructions](/benchmarks//recommendation/tensorflow/wide_deep_large_ds/README.md#fp32-inference-instructions) to download the Intel optimized
Wide and Deep pre-trained model graph.

The following steps show how to convert the `FP32` model to `Int8` precision to reduce the model size:

6. Convert the FP32-graph to a dynamic range Int8-graph using the output node names (from step 1)

    ```
        $ python tensorflow/tools/quantization/quantize_graph.py \
        --input=/workspace/quantization/optimized_wide_deep_fp32_graph.pb \
        --output=/workspace/quantization/int8_dynamic_range_wide_deep_graph.pb \
        --output_node_names='import/head/predictions/probabilities' \
        --mode=eightbit \
        --intel_cpu_eightbitize=True \
        --model_name=wide_deep_large_ds
    ```

    [Evaluate the output int8 graph performance](#accuracy-for-int8-optimized-graph)
    to check the loss in performance after the model quantization.

7. Convert from dynamic to static re-quantization range.
The following steps are to freeze the re-quantization range:

    * Insert the logging op:
        ```
            $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=/workspace/quantization/int8_dynamic_range_wide_deep_graph.pb \
            --out_graph=/workspace/quantization/logged_int8_dynamic_range_wide_deep.pb \
            --transforms='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'
        ```

        * **Generate the `wide_deep_min_max_log.txt` file**, follow [instructions](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/README.md#int8-inference-instructions)
            to run inference (using `--batch_size=1024`,
            `--data-location=/home/<user>/dataset_preprocessed_train.tfrecords`refer [instructions](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/README.md#prepare-dataset) to generate training data,
            `--in-graph=/home/<user>/logged_int8_dynamic_range_wide_deep.pb`,
            `--accuracy-only`), and **store the output log in `wide_deep_min_max_log.txt` file**.

        * The `wide_deep_min_max_log.txt` file is used in the following step.

    * Run the log data replace the

        `RequantizationRangeOp` with constants in the original quantized graph:
        ```
            $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=/workspace/quantization/int8_dynamic_range_wide_deep_graph.pb \
            --out_graph=/workspace/quantization/freezed_range_int8_wide_deep.pb \
            --transforms='freeze_requantization_ranges(min_max_log_file="/workspace/quantization/wide_deep_min_max_log.txt")'
        ```

         [Evaluate the output int8 graph performance](#accuracy-for-int8-optimized-graph)
    to check the loss in performance after this step.

8. Convert from dynamic to static quantization Min range.The following steps are to freeze the Min range:

    * Insert the logging op:
        ```
            $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=/workspace/quantization/freezed_range_int8_wide_deep.pb \
            --out_graph=/workspace/quantization/logged_freezed_range_int8_wide_deep.pb \
            --transforms='insert_logging(op=Min, show_name=true, message="__min:")'
        ```

        * **Generate the `wide_deep_min_log.txt` file**, follow [instructions](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/README.md#int8-inference-instructions)
            to run inference (using `--batch_size=1024`,
            `--data-location=/home/<user>/dataset_preprocessed_train.tfrecords`refer [instructions](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/README.md#prepare-dataset) to generate training data,
            `--in-graph=/home/<user>/logged_freezed_range_int8_wide_deep.pb`,
            `--accuracy-only`), and **store the output log in `wide_deep_min_log.txt` file**.

        * The `wide_deep_min_log.txt` file is used in the following step.

    * Run the log data replace the

        `MinOp` with constants in the original quantized graph:
        ```
            $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=/workspace/quantization/freezed_range_int8_wide_deep.pb \
            --out_graph=/workspace/quantization/freezed_range_int8_wide_deep_freezemin.pb \
            --transforms='freeze_min(min_max_log_file="/workspace/quantization/wide_deep_min_log.txt")'
        ```

         [Evaluate the output int8 graph performance](#accuracy-for-int8-optimized-graph)
    to check the loss in performance after this step.

9. Convert from dynamic to static quantization Max range.
The following steps are to freeze the Max range:

    * Insert the logging op:
        ```
            $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=/workspace/quantization/freezed_range_int8_wide_deep.pb \
            --out_graph=/workspace/quantization/logged_freezed_range_int8_wide_deep.pb \
            --transforms='insert_logging(op=Max, show_name=true, message="__max:")'
        ```

        * **Generate the `wide_deep_max_log.txt` file**, follow [instructions](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/README.md#int8-inference-instructions)
            to run inference (using `--batch_size=1024`,
            `--data-location=/home/<user>/dataset_preprocessed_train.tfrecords`refer [instructions](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/README.md#prepare-dataset) to generate training data,
            `--in-graph=/home/<user>/logged_freezed_range_int8_wide_deep.pb`,
            `--accuracy-only`), and **store the output log in `wide_deep_max_log.txt` file**.

        * The `wide_deep_max_log.txt` file is used in the following step.

    * Run the log data replace the

        `MaxOp` with constants in the original quantized graph:
        ```
            $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=/workspace/quantization/freezed_range_int8_wide_deep_freezemin.pb \
            --out_graph=/workspace/quantization/freezed_range_int8_wide_deep_minmaxfreeze.pb \
            --transforms='freeze_max(min_max_log_file="/workspace/quantization/wide_deep_max_log.txt")'
        ```

         [Evaluate the output int8 graph performance](#accuracy-for-int8-optimized-graph)
    to check the loss in performance after this step.

10. Fuse `RequantizeOp` with fused quantized innerproducts, and generate the final
optimized Int8 graph
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=/workspace/quantization/freezed_range_int8_wide_deep_minmaxfreeze.pb \
        --out_graph=/workspace/quantization/final_int8_wide_deep.pb \
        --outputs='import/head/predictions/probabilities' \
        --transforms='fuse_quantized_matmul_and_requantize strip_unused_nodes'
    ```
    Check the final quantized wide and deep model `final_int8_wide_deep.pb` performance in
    the [Accuracy for Int8 Optimized Graph](#accuracy-for-int8-optimized-graph) section.


## Performance Evaluation

Validating the model performance is required after each step to verify if the output graph achieves the accuracy target.
* The model accuracy is used as a performance measure.
* The accuracy target is the optimized FP32 model accuracy values.
* The quantized `Int8` graph accuracy should not drop more than ~0.5-1%.


This section explains how to run wide & deep inference and calculate the model accuracy using the [Intel Model Zoo](https://github.com/IntelAI/models).

Clone the [IntelAI/models](https://github.com/IntelAI/models) repository,
and follow the [documented steps](/benchmarks/recommendation/tensorflow/wide_deep_large_ds/README.md)
to run `Wide and Deep` inference performance for both FP32 and Int8 cases.

**Note that the script should be run outside of the quantization docker container
and that some inputs to the script are slightly different for `FP32` and `Int8` models (i.e. `--precision` and `--docker-image`).**


### Accuracy for FP32 Optimized Graph
Clone the [IntelAI/models](https://github.com/IntelAI/models) repository and follow the steps to run the FP32
script to calculate `accuracy` and use the optimized FP32 graph in `--in-graph`.
   ```
        $ git clone https://github.com/IntelAI/models.git
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/<pretrained_model_directory>/wide_deep_fp32_pretrained_model.pb \
            --model-name wide_deep_large_ds \
            --framework tensorflow \
            --precision fp32 \
            --mode inference \
            --accuracy-only \
            --batch-size=1000 \
            --socket-id 0 \
            --data-location /root/user/wide_deep_files/dataset_preprocessed_eval.tfrecords \
            --docker-image  docker.io/intelaipg/intel-optimized-tensorflow:latest
  ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
   --------------------------------------------------
   Total test records           :  2000000
   Batch size is                :  512
   Number of batches            :  3907
   Throughput is (records/sec)  :  314943.875
   Inference duration (seconds) :  6.1878
   Latency (millisecond/batch)  :  1.625686
   Classification accuracy (%)  :  77.5223
   No of correct predicitons    :  1550447
   --------------------------------------------------
   ```

### Accuracy for Int8 Optimized Graph

Clone the [IntelAI/models](https://github.com/IntelAI/models) repository and follow the steps to run the Int8
script to calculate `accuracy` and use the Int8 graph in `--in-graph`.
   ```
        $ git clone https://github.com/IntelAI/models.git
        $ cd /home/<user>/models/benchmarks
        $ python launch_benchmark.py \
            --in-graph /home/<user>/<pretrained_model_directory>/final_wide_deep_Int8_graph.pb \
            --model-name wide_deep_large_ds \
            --framework tensorflow \
            --precision int8 \
            --mode inference \
            --accuracy-only \
            --batch-size=1000 \
            --socket-id 0 \
            --data-location /home/<user>/<dataset_directory>/dataset_preprocessed_eval.tfrecords \
            --docker-image  docker.io/intelaipg/intel-optimized-tensorflow:latest
   ```
The tail of the log output when the accuracy run completes should look something like this:
   ```
   --------------------------------------------------
   Total test records           :  2000000
   Batch size is                :  512
   Number of batches            :  3907
   Throughput is (records/sec)  :  489653.313
   Inference duration (seconds) :  3.98
   Latency (millisecond/batch)  :  1.045638
   Classification accuracy (%)  :  77.4816
   No of correct predicitons    :  1549632
   --------------------------------------------------
   ```
