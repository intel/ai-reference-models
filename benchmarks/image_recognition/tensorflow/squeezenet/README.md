# SqueezeNet

This document has instructions for how to run SqueezeNet for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model training and inference for
other precisions are coming later.

## FP32 Inference Instructions

1. Store the path to the current directory and clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
$ pushd $MODEL_WORK_DIR

$ git clone https://github.com/IntelAI/models.git
```

This repository includes launch scripts for running SqueezeNet,
checkpoint files for restoring a pre-trained SqueezeNet model, and
CPU optimized SqueezeNet model scripts.

2. Register and download the
[ImageNet dataset](http://image-net.org/download-images).

3. Once you have the raw ImageNet dataset downloaded, we need to convert
it to the TFRecord format.  This is done using the
[build_imagenet_data.py](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py)
script.  There are instructions in the header of the script explaining
its usage.

After the script has completed, you should have a directory with the
sharded dataset something like:

```
$ ll $MODEL_WORK_DIR/datasets/ImageNet_TFRecords
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

4. Download and extract the pretrained model and then store the path to the current directory:
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/squeezenet_fp32_pretrained_model.tar.gz
$ tar -xvf squeezenet_fp32_pretrained_model.tar.gz
```

5. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.

```
$ cd $MODEL_WORK_DIR/models/benchmarks
```

The `launch_benchmark.py` script in the `benchmarks` directory is used
for starting a model run in a TensorFlow docker container. It has
arguments to specify which model, framework, mode, precision, and docker
image to use, along with your path to the ImageNet TF Records that you
generated in step 3 and the checkpoint files that you downloaded in
step 4.

Substitute in your own `--data-location` and follow the steps in the
following example for batch inference (using `--batch-size 64`):

```
$ python launch_benchmark.py \
    --precision fp32 \
    --model-name squeezenet \
    --mode inference \
    --framework tensorflow \
    --socket-id 0 \
    --batch-size 64 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --checkpoint $MODEL_WORK_DIR/squeezenet_checkpoints \
    --data-location $MODEL_WORK_DIR/datasets/ImageNet_TFRecords
```

Or for online inference (using `--batch-size 1`):

```
$ python launch_benchmark.py \
    --precision fp32 \
    --model-name squeezenet \
    --mode inference \
    --framework tensorflow \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --checkpoint $MODEL_WORK_DIR/squeezenet_checkpoints \
    --data-location $MODEL_WORK_DIR/datasets/ImageNet_TFRecords
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

6. The log file is saved to the value of `--output-dir`.

The tail of the log output when the script completes should look
something like this, when running for batch inference with `--batch-size 64`:

```
SqueezeNet Inference Summary:
            250 batches x 64 bs = total 16000 images evaluated
            batch size = 64
            throughput[med] = 837.1 image/sec
            latency[median] = 1.195 ms

Ran inference with batch size 64
Log location outside container: {--output-dir value}/benchmark_squeezenet_inference_fp32_20190104_220051.log
```

Or for online inference (with `--batch-size 1`):

```
SqueezeNet Inference Summary:
            250 batches x 1 bs = total 250 images evaluated
            batch size = 1
            throughput[med] = 115.3 image/sec
            latency[median] = 8.67 ms

Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_squeezenet_inference_fp32_20190104_220712.log
```

7. To return to where you started from:
```
$ popd
```
