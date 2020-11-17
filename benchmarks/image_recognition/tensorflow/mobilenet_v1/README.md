# MobileNet V1

This document has instructions for how to run MobileNet V1 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)
* [BFloat16 inference](#bfloat16-inference-instructions)

Instructions and scripts for model training are coming
later.


## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Download ImageNet dataset.

    This step is required only for running accuracy, for running benchmark we do not need to provide dataset.

    Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
    After running the conversion script you should have a directory with the
    ImageNet dataset in the TF records format.

2. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/mobilenetv1_int8_pretrained_model.pb
```

3. Clone the [intelai/models](https://github.com/intelai/models) repo
    and then run the model scripts for either online or batch inference or accuracy. For --dataset-location in accuracy run, please use the ImageNet validation data path from step 1.
    Each model run has user configurable arguments separated from regular arguments by '--' at the end of the command.
    Unless configured, these arguments will run with default values. Below are the example codes for each use case:
    
    ```
    $ git clone https://github.com/IntelAI/models.git

    $ cd benchmarks
    ```

    For batch inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 240`):
    ```
    python launch_benchmark.py  \
         --model-name mobilenet_v1 \
         --precision int8 \
         --mode inference \
         --framework tensorflow \
         --benchmark-only \
         --batch-size 240  \
         --socket-id 0 \
         --in-graph /home/<user>/mobilenetv1_int8_pretrained_model.pb  \
         --docker-image intel/intel-optimized-tensorflow:2.3.0 \
         -- input_height=224 input_width=224 warmup_steps=10 steps=50 \
         input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
    ```

    For online inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`)
    ```
    python launch_benchmark.py  \
         --model-name mobilenet_v1 \
         --precision int8 \
         --mode inference \
         --framework tensorflow \
         --benchmark-only \
         --batch-size 1  \
         --socket-id 0 \
         --in-graph /home/<user>/mobilenetv1_int8_pretrained_model.pb  \
         --docker-image intel/intel-optimized-tensorflow:2.3.0 \
         -- input_height=224 input_width=224 warmup_steps=10 steps=50 \
         input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
    ```

    For accuracy (using your `--data-location`, `--accuracy-only` and
    `--batch-size 100`):
    ```
    python launch_benchmark.py  \
         --model-name mobilenet_v1 \
         --precision int8 \
         --mode inference \
         --framework tensorflow \
         --accuracy-only \
         --batch-size 100  \
         --socket-id 0 \
         --in-graph /home/<user>/mobilenetv1_int8_pretrained_model.pb  \
         --docker-image intel/intel-optimized-tensorflow:2.3.0 \
         --data-location /home/<user>/imagenet_validation_dataset \
         -- input_height=224 input_width=224 \
         input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
    ```

    Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
    to get additional debug output or change the default output location.

    At present no relesed docker image support the latest MobileNet Int8 inference and accuracy.

4. The log file is saved to the `models/benchmarks/common/tensorflow/logs` directory,
    or the directory specified by the `--output-dir` arg. Below are examples of
    what the tail of your log file should look like for the different configs.

    Example log tail when running for batch inference:
    ```
    [Running warmup steps...]
    steps = 10, 1865.30956528 images/sec
    [Running benchmark steps...]
    steps = 10, 1872.92398031 images/sec
    steps = 20, 1862.64499512 images/sec
    steps = 30, 1857.97283454 images/sec
    steps = 40, 1864.70142784 images/sec
    steps = 50, 1854.23896906 images/sec
    Ran inference with batch size 240
    Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190523_164626.log
    ```

    Example log tail when running for online inference:
    ```
    [Running warmup steps...]
    steps = 10, 197.082229114 images/sec
    [Running benchmark steps...]
    steps = 10, 195.201936054 images/sec
    steps = 20, 195.693743293 images/sec
    steps = 30, 198.999098543 images/sec
    steps = 40, 189.256565292 images/sec
    steps = 50, 201.252531069 images/sec
    Ran inference with batch size 1
    Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190523_164348.log
    ```

    Example log tail when running for accuracy:
    ```
    Iteration time: 66.8541 ms
    Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7014, 0.8935)
    Iteration time: 66.7909 ms
    Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7014, 0.8934)
    Iteration time: 66.7001 ms
    Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7014, 0.8934)
    Ran inference with batch size 100
    Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190523_164955.log
    ```

## FP32 Inference Instructions

1. The ImageNet dataset is required for testing accuracy and can also be
   used when running online or batch inference. If no dataset is provided when running
   online or batch inference, synthetic data will be used.

   Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
   After running the conversion script you should have a directory with the
   ImageNet dataset in the TF records format.

2. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/mobilenet_v1_1.0_224_frozen.pb
```

3. Clone the [tensorflow/models](https://github.com/tensorflow/models)
   repository.

    ```
    $ git clone https://github.com/tensorflow/models
    ```

    The [tensorflow/models](https://github.com/tensorflow/models) files
    are used for dependencies when running the model.

4. Clone the [intelai/models](https://github.com/IntelAI/models) repo
   and then navigate to the benchmarks directory:

   ```
   $ git clone https://github.com/IntelAI/models.git
   $ cd models/benchmarks
   ```

   MobileNet V1 can be run for either online or batch inference using the
   commands below.  The `--data-location` should be the path to the
   ImageNet validation data from step 1, the `--checkpoint` arg should
   be the path to the checkpoint files from step 2, and the
   `--model-source-dir` should point to the
   [tensorflow/models](https://github.com/tensorflow/models) repo that
   was cloned in step 3.

   * Run for online inference (with `--batch-size 1`, `--checkpoint`
     with a path to the checkpoint file directory, and the `--data-location`
     is optional):
     
     ```
     python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intel/intel-optimized-tensorflow:2.3.0 \
         --model-source-dir /home/<user>/tensorflow/models  \
         --batch-size 1 \
         --socket-id 0 \
         --data-location /dataset/Imagenet_Validation \
         --in-graph /home/<user>/mobilenet_v1_1.0_224_frozen.pb \
         -- input_height=224 input_width=224 warmup_steps=10 steps=50 \
         input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
     ```

    * Run for batch inference (with `--batch-size 100`,
      `--checkpoint` with a path to the checkpoint file directory, and
      the `--data-location` is optional):

      ```
      python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intel/intel-optimized-tensorflow:2.3.0 \
         --model-source-dir /home/<user>/tensorflow/models  \
         --batch-size 100 \
         --socket-id 0 \
         --data-location /dataset/Imagenet_Validation \
         --in-graph /home/<user>/mobilenet_v1_1.0_224_frozen.pb \
         -- input_height=224 input_width=224 warmup_steps=10 steps=50 \
         input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
      ```
    * Run for accuracy (with `--batch-size 100`, `--accuracy-only` and `--in-graph` with a path to the frozen graph .pb file):
      ```
      python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intel/intel-optimized-tensorflow:2.3.0 \
         --model-source-dir /home/<user>/tensorflow/models  \
         --batch-size 100 \
         --accuracy-only \
         --data-location /dataset/Imagenet_Validation \
         --in-graph /home/<user>/mobilenet_v1_1.0_224_frozen.pb
      ```
      Note that the `--verbose` or `--output-dir` flag can be added to any of the above
      commands to get additional debug output or change the default output location.

5. The log files for each run are saved at the value of `--output-dir`.

   * Below is a sample log file snippet when testing online inference:
     ```
     2019-01-04 20:02:23.855441: step 80, 78.3 images/sec
     2019-01-04 20:02:23.974862: step 90, 83.7 images/sec
     2019-01-04 20:02:24.097476: step 100, 84.0 images/sec
     eval/Accuracy[0]
     eval/Recall_5[0]
     INFO:tensorflow:Finished evaluation at 2019-01-04-20:02:24
     self._total_images_per_sec = 809.6
     self._displayed_steps = 10
     Total images/sec = 81.0
     Latency ms/step = 12.4
     Ran inference with batch size 1
     Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_fp32_20190104_200218.log
     ```

   * Below is a sample log file snippet when testing batch inference:
     ```
     2019-01-04 20:06:01.151312: step 80, 184.0 images/sec
     2019-01-04 20:06:06.719081: step 90, 180.5 images/sec
     2019-01-04 20:06:12.346302: step 100, 174.1 images/sec
     eval/Accuracy[0.0009]
     eval/Recall_5[0.0049]
     INFO:tensorflow:Finished evaluation at 2019-01-04-20:06:12
     self._total_images_per_sec = 1810.2
     self._displayed_steps = 10
     Total images/sec = 181.0
     Ran inference with batch size 100
     Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_fp32_20190104_200512.log
     ```
   * Below is a sample lof file snippet when testing accuracy:
     ```
     Iteration time: 119.1134 ms
     Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7104, 0.8999)
     Iteration time: 118.8375 ms
     Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7103, 0.8999)
     Iteration time: 119.9311 ms
     Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7102, 0.8999)
     Ran inference with batch size 100
     Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_fp32_20190110_211648.log
     ```

# BFloat16 Inference Instructions

MobileNet v1 BFloat16 inference depends on Auto-Mixed-Precision to convert graph from FP32 to BFloat16 online.
Before evaluating MobileNet v1 BFloat16 inference, please set the following environment variables:

```
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_REMOVE=BiasAdd \
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE=Softmax \
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD=BiasAdd,Softmax
```

The instructions are the same as FP32 inference instructions above, except one needs to change the `--precision=fp32` to `--precision=bfloat16` in the above commands.
