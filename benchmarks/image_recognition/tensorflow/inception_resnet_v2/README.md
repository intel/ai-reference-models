# Inception ResNet V2

This document has instructions for how to run Inception ResNet V2 for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone git@github.com:IntelAI/models.git
```

This repository includes launch scripts for running benchmarks and the
an optimized version of the Inception ResNet V2 model code.

2. Download the pre-trained Inception ResNet V2 model files:

For accuracy:

A link to download the pre-trained model is coming soon.

For throughput and latency:

```
$ wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
$ mkdir -p checkpoints && tar -C ./checkpoints/ -zxf inception_resnet_v2_2016_08_30.tar.gz
```

3. If you would like to run Inception ResNet V2 inference and test for
accuracy, you will need the full ImageNet dataset. Benchmarking for latency
and throughput do not require the ImageNet dataset.

Register and download the
[ImageNet dataset](http://image-net.org/download-images).

Once you have the raw ImageNet dataset downloaded, we need to convert
it to the TFRecord format. This is done using the
[build_imagenet_data.py](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py)
script. There are instructions in the header of the script explaining
its usage.

After the script has completed, you should have a directory with the
sharded dataset something like:

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

4. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the ImageNet
TF Records that you generated in step 3.

Substitute in your own `--data-location` (from step 3, for accuracy
only), `--checkpoint` pre-trained model checkpoint file path (from step 2).

Inception ResNet V2 can be run for accuracy, latency benchmarking, or throughput
benchmarking. Use one of the following examples below, depending on
your use case.

For accuracy (using your `--data-location`, `--accuracy-only` and
`--batch-size 100`):

```
python launch_benchmark.py \
    --model-name inception_resnet_v2 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --in-graph /home/myuser/inception_resnet_v2_int8_pretrained_model.pb \
    --data-location /home/myuser/datasets/ImageNet_TFRecords
```

For latency (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name inception_resnet_v2 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --socket-id 0 \
    --checkpoint /home/myuser/checkpoints \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --data-location /home/myuser/datasets/ImageNet_TFRecords
```

For throughput (using `--benchmark-only`, `--socket-id 0` and `--batch-size 128`):

```
python launch_benchmark.py \
    --model-name inception_resnet_v2 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 128 \
    --socket-id 0 \
    --checkpoint /home/myuser/checkpoints \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --data-location /home/myuser/datasets/ImageNet_TFRecords
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..

6. The log file is saved to the value
of `--output-dir`. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for accuracy:

```
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.8036, 0.9526)
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.8036, 0.9525)
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.8037, 0.9525)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_inception_resnet_v2_inference_fp32_20190109_081637.log
```

Example log tail when benchmarking for latency:
```
eval/Accuracy[0]
eval/Recall_5[0.01]
INFO:tensorflow:Finished evaluation at 2019-01-08-01:51:28
self._total_images_per_sec = 69.7
self._displayed_steps = 10
Total images/sec = 7.0
Latency ms/step = 143.4
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_inception_resnet_v2_inference_fp32_20190108_015057.log
```

Example log tail when benchmarking for throughput:
```
eval/Accuracy[0.00078125]
eval/Recall_5[0.00375]
INFO:tensorflow:Finished evaluation at 2019-01-08-01:59:37
self._total_images_per_sec = 457.0
self._displayed_steps = 10
Total images/sec = 45.7
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_inception_resnet_v2_inference_fp32_20190108_015440.log

