# Inception V4

This document has instructions for how to run Inception V4 for the
following modes/platforms:
* [Int8 inference](#int8-inference-instructions)

Benchmarking instructions and scripts for model training and inference
other platforms are coming later.

## Int8 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone git@github.com:IntelAI/models.git
```

This repository includes launch scripts for running benchmarks.

2. Download the pre-trained Inception V4 Int8 model:

```
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/inceptionv4_int8_pretrained_model.pb
```

3. Build a docker image using master of the official
[TensorFlow](https://github.com/tensorflow/tensorflow) repository with
`--config=mkl`. More instructions on
[how to build from source](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#inpage-nav-5).

4. If you would like to run Inception V4 inference and test for
accuracy, you will need the ImageNet dataset. Benchmarking for latency
and throughput do not require the ImageNet dataset.  Instructions for
downloading the ImageNet dataset and converting it to the TF Records
format and be found
[here](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data).

5. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
platform, and docker image to use, along with your path to the ImageNet
TF Records that you generated in step 4.

Inception V4 can be run to test accuracy or benchmarking throughput or
latency. Use one of the following examples below, depending on your use
case.

For accuracy (using your `--data-location`, `--accuracy-only` and
`--batch-size 100`):

```
python launch_benchmark.py \
    --model-name inceptionv4 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --single-socket \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/inceptionv4_int8_pretrained_model.pb \
    --data-location /home/myuser/datasets/ImageNet_TFRecords
```

For throughput benchmarking (using `--benchmark-only`, `--single-socket` and `--batch-size 240`):

```
python launch_benchmark.py \
    --model-name inceptionv4 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 240 \
    --single-socket \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/inceptionv4_int8_pretrained_model.pb
```

For latency (using `--benchmark-only`, `--single-socket` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name inceptionv4 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --single-socket \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/inceptionv4_int8_pretrained_model.pb
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

7. The log file is saved to the
`intelai/models/benchmarks/common/tensorflow/logs` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for accuracy:

```
Processed 48960 images. (Top1 accuracy, Top5 accuracy) = (0.7994, 0.9505)
Processed 49200 images. (Top1 accuracy, Top5 accuracy) = (0.7995, 0.9505)
Processed 49440 images. (Top1 accuracy, Top5 accuracy) = (0.7994, 0.9504)
Processed 49680 images. (Top1 accuracy, Top5 accuracy) = (0.7996, 0.9505)
Processed 49920 images. (Top1 accuracy, Top5 accuracy) = (0.7996, 0.9503)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 240
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_inceptionv4_inference_int8_20181212_210600.log
```

Example log tail when benchmarking for throughput:
```
[Running warmup steps...]
steps = 10, 175.298147727 images/sec
[Running benchmark steps...]
steps = 10, 174.898030455 images/sec
steps = 20, 175.043739505 images/sec
steps = 30, 174.010253581 images/sec
steps = 40, 174.023128794 images/sec
steps = 50, 174.824917692 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 240
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_inceptionv4_inference_int8_20181212_210320.log
```

Example log tail when benchmarking for latency:
```
[Running warmup steps...]
steps = 10, 21.5271353637 images/sec
[Running benchmark steps...]
steps = 10, 20.4683066803 images/sec
steps = 20, 15.3893434503 images/sec
steps = 30, 22.8806842978 images/sec
steps = 40, 22.1703834871 images/sec
steps = 50, 21.2602339774 images/sec
Latency: 46.514 ms
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_inceptionv4_inference_int8_20181215_000616.log
```
