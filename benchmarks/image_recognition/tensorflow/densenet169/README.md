# DenseNet 169

This document has instructions for how to run DenseNet 169 for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions

1. Download ImageNet dataset.

    This step is required only for running accuracy, for running the model for performance we do not need to provide dataset.

    Register and download the ImageNet dataset. Once you have the raw ImageNet dataset downloaded, we need to convert
    it to the TFRecord format. The TensorFlow models repo provides
    [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
    to download, process and convert the ImageNet dataset to the TF records format. After converting data, you should have a directory
    with the sharded dataset something like below, we only need `validation-*` files, discard `train-*` files:
    ```
    $ ll /home/myuser/datasets/ImageNet_TFRecords
    -rw-r--r--. 1 user 143009929 Jun 20 14:53 train-00000-of-01024
    -rw-r--r--. 1 user 144699468 Jun 20 14:53 train-00001-of-01024
    -rw-r--r--. 1 user 138428833 Jun 20 14:53 train-00002-of-01024
    ...
    -rw-r--r--. 1 user 143137777 Jun 20 15:08 train-01022-of-01024
    -rw-r--r--. 1 user 143315487 Jun 20 15:08 train-01023-of-01024
    -rw-r--r--. 1 user  52223858 Jun 20 15:08 validation-00000-of-00128
    -rw-r--r--. 1 user  51019711 Jun 20 15:08 validation-00001-of-00128
    -rw-r--r--. 1 user  51520046 Jun 20 15:08 validation-00002-of-00128
    ...
    -rw-r--r--. 1 user  52508270 Jun 20 15:09 validation-00126-of-00128
    -rw-r--r--. 1 user  55292089 Jun 20 15:09 validation-00127-of-00128
    ```

2. Download the pretrained model:
   ```
   $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/densenet169_fp32_pretrained_model.pb
   ```

3. Clone the [intelai/models](https://github.com/intelai/models) repo
    and then run the model scripts for either online or batch inference or accuracy. For --dataset-location in accuracy run, please use the ImageNet validation data path from step 1.
    Each model run has user configurable arguments separated from regular arguments by '--' at the end of the command.
    Unless configured, these arguments will run with default values. Below are the example codes for each use case:

    ```
    $ git clone https://github.com/IntelAI/models.git

    $ cd benchmarks
    ```

    For throughput (using `--benchmark-only`, `--socket-id 0` and `--batch-size 100`):
    ```
    python launch_benchmark.py \
        --model-name densenet169 \
        --precision fp32 \
        --mode inference \
        --framework tensorflow \
        --benchmark-only \
        --batch-size 100 \
        --socket-id 0 \
        --in-graph /home/<user>/densenet169_fp32_pretrained_model.pb \
        --docker-image intel/intel-optimized-tensorflow:2.1.0 \
        -- input_height=224 input_width=224 warmup_steps=20 steps=100 \
        input_layer="input" output_layer="densenet169/predictions/Reshape_1"
    ```

    For latency (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`)
    ```
    python launch_benchmark.py \
        --model-name densenet169 \
        --precision fp32 \
        --mode inference \
        --framework tensorflow \
        --benchmark-only \
        --batch-size 1 \
        --socket-id 0 \
        --in-graph /home/<user>/densenet169_fp32_pretrained_model.pb \
        --docker-image intel/intel-optimized-tensorflow:2.1.0 \
        -- input_height=224 input_width=224 warmup_steps=20 steps=100 \
        input_layer="input" output_layer="densenet169/predictions/Reshape_1"
    ```

    For accuracy (using your `--data-location`, `--socket-id 0`, `--accuracy-only` and
    `--batch-size 100`):
    ```
    python launch_benchmark.py \
        --model-name densenet169 \
        --precision fp32 \
        --mode inference \
        --framework tensorflow \
        --accuracy-only \
        --batch-size 100 \
        --socket-id 0 \
        --in-graph /home/<user>/densenet169_fp32_pretrained_model.pb \
        --docker-image intel/intel-optimized-tensorflow:2.1.0 \
        --data-location /home/<user>/imagenet_validation_dataset \
        -- input_height=224 input_width=224 \
        input_layer="input" output_layer="densenet169/predictions/Reshape_1"
    ```

    Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
    to get additional debug output or change the default output location.

4. The log file is saved to the `models/benchmarks/common/tensorflow/logs` directory,
    or the directory specified by the `--output-dir` arg. Below are examples of
    what the tail of your log file should look like for the different configs.

    Example log tail when running for batch inference:
    ```
    steps = 80, 159.83471377 images/sec
           Latency: 625.646317005 ms
    steps = 90, 159.852789241 images/sec
           Latency: 625.57557159 ms
    steps = 100, 159.853966416 images/sec
           Latency: 625.570964813 ms
    Ran inference with batch size 100
    Log location outside container: {--output-dir value}/benchmark_densenet169_inference_fp32_20190412_023940.log
    ```

    Example log tail when running for online inference:
    ```
    steps = 80, 34.9948442873 images/sec
           Latency: 28.5756379366 ms
    steps = 90, 34.9644341907 images/sec
           Latency: 28.6004914178 ms
    steps = 100, 34.9655988121 images/sec
           Latency: 28.5995388031 ms
    Ran inference with batch size 1
    Log location outside container: {--output-dir value}/benchmark_densenet169_inference_fp32_20190412_024505.log
    ```

    Example log tail when running for accuracy:
    ```
    Iteration time: 581.6446 ms
    0.757505030181
    Iteration time: 581.5755 ms
    0.757489959839
    Iteration time: 581.5709 ms
    0.75749498998
    Iteration time: 581.1705 ms
    0.75748
    Ran inference with batch size 100
    Log location outside container: {--output-dir value}/benchmark_densenet169_inference_fp32_20190412_021545.log
    ```
