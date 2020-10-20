# Inception V3

This document has instructions for how to run Inception V3 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Instructions for model training and inference for other precisions are coming later.

## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

This repository includes launch scripts for running an optimized version of the Inception V3 model code.

2. Clone the [tensorflow/models](https://github.com/tensorflow/models)
repository as `tensorflow-models`. This is to avoid conflict with Intel's `models` repo:

```
git clone https://github.com/tensorflow/models.git tensorflow-models
```

This repository is used for dependencies that the Inception V3 model
requires.

3. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv3_int8_pretrained_model.pb
```

4. If you would like to run Inception V3 inference with real data or test for
accuracy, you will need the ImageNet dataset.

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

5. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the ImageNet
TF Records that you generated in step 4.

Substitute in your own `--data-location` (from step 4, for accuracy
only), `--in-graph` pretrained model file path (from step 3) and
`--model-source-dir` for the location where you cloned the
[tensorflow/models](https://github.com/tensorflow/models) repo
(from step 2).

Inception V3 can be run for accuracy, online inference, or batch inference. 
Use one of the following examples below, depending on your use case.

For accuracy (using your `--data-location`, `--accuracy-only` and
`--batch-size 100`):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --in-graph /home/<user>/inceptionv3_int8_pretrained_model.pb \
    --data-location /home/<user>/datasets/ImageNet_TFRecords
```

When testing performance, it is optional to specify the
number of `warmup_steps` and `steps` as extra args, as shown in the
commands below. If these values are not specified, the script will
default to use `warmup_steps=10` and `steps=50`.

For online inference with ImageNet data (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --in-graph /home/<user>/inceptionv3_int8_pretrained_model.pb \
    --data-location /home/<user>/datasets/ImageNet_TFRecords \
    -- warmup_steps=50 steps=500
```

For online inference with dummy data (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`), remove `--data-location` argument:

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --in-graph /home/<user>/inceptionv3_int8_pretrained_model.pb \
    -- warmup_steps=50 steps=500
```

For batch inference with ImageNet data (using `--benchmark-only`, `--socket-id 0` and `--batch-size 128`):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 128 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --in-graph /home/<user>/inceptionv3_int8_pretrained_model.pb \
    --data-location /home/<user>/datasets/ImageNet_TFRecords \
    -- warmup_steps=50 steps=500
```

For batch inference with dummy data (using `--benchmark-only`, `--socket-id 0` and `--batch-size 128`), remove `--data-location` argument::

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 128 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --in-graph /home/<user>/inceptionv3_int8_pretrained_model.pb \
    -- warmup_steps=50 steps=500
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..

7. The log file is saved to the value
of `--output-dir`. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for accuracy:

```
Iteration time: 357.3781 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7666, 0.9333)
Executing command: python /workspace/intelai_models/int8/accuracy.py --input_height=299 --input_width=299 --num_intra_threads=56 --num_inter_threads=2 --batch_size=100 --input_graph=/in_graph/inceptionv3_int8_pretrained_model.pb --data_location=/dataset
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_inceptionv3_inference_int8_20190104_013246.log
```

Example log tail when running for online inference:
```
...
steps = 470, 134.912798739 images/sec
steps = 480, 132.379245045 images/sec
steps = 490, 133.977640069 images/sec
steps = 500, 132.083262478 images/sec
Average throughput for batch size 1: 133.440858806 images/sec
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_inceptionv3_inference_int8_20190415_220455.log
```

Example log tail when running for batch inference:
```
...
steps = 470, 369.151656047 images/sec
steps = 480, 373.174541014 images/sec
steps = 490, 372.402638382 images/sec
steps = 500, 371.836748659 images/sec
Average throughput for batch size 128: 371.269087408 images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_inceptionv3_inference_int8_20190416_162155.log
```

## FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

2. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv3_fp32_pretrained_model.pb
```

3. If you would like to run Inception V3 FP32 inference and test for
accuracy, you will need the ImageNet dataset.

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

4. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image.

Substitute in your own `--in-graph` pretrained model file path (from step 2).

Inception V3 can be run for online inference, batch inference, or accuracy. Use one of the following examples below,
depending on your use case.

* For online inference with dummy data (using `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --in-graph /home/<user>/inceptionv3_fp32_pretrained_model.pb
```
Example log tail when running for online inference:
```
Inference with dummy data.
Iteration 1: 1.075 sec
Iteration 2: 0.023 sec
Iteration 3: 0.016 sec
...
Iteration 38: 0.014 sec
Iteration 39: 0.014 sec
Iteration 40: 0.014 sec
Average time: 0.014 sec
Batch size = 1
Latency: 14.442 ms
Throughput: 69.243 images/sec
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_inceptionv3_inference_fp32_20190104_025220.log
```

* For batch inference with dummy data (using `--batch-size 128`):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 128 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --in-graph /home/<user>/inceptionv3_fp32_pretrained_model.pb
```
Example log tail when running for batch inference:
```
Inference with dummy data.
Iteration 1: 2.024 sec
Iteration 2: 0.765 sec
Iteration 3: 0.781 sec
...
Iteration 38: 0.756 sec
Iteration 39: 0.760 sec
Iteration 40: 0.757 sec
Average time: 0.760 sec
Batch size = 128
Throughput: 168.431 images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_inceptionv3_inference_fp32_20190104_024842.log
```

* For accuracy (using `--accuracy-only`, `--batch-size 100`, and
`--data-location` with the path to the ImageNet dataset from step 3):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --data-location /dataset/Imagenet_Validation \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --in-graph /home/<user>/inceptionv3_fp32_pretrained_model.pb
```
Example log tail when running for accuracy:
```
Iteration time: 756.7571 ms
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7673, 0.9341)
Iteration time: 757.3781 ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7674, 0.9341)
Iteration time: 760.3024 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7675, 0.9342)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_inceptionv3_inference_fp32_20190104_023816.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..
