# SqueezeNet

This document has instructions for how to run SqueezeNet for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference
other platforms are coming later.

## FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone git@github.com:IntelAI/models.git
```

This repository includes launch scripts for running benchmarks,
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
the [intelai/models](https://github.com/IntelAI/models) repo from step
1, and get the path the to squeezenet checkpoint files:

```
$ cd /home/myuser/models/benchmarks

$ checkpoint_dir=$(pwd)/image_recognition/tensorflow/squeezenet/inference/fp32/checkpoints
```

The `launch_benchmark.py` script in the `benchmarks` directory is used
for starting a benchmarking run in a TensorFlow docker container. It has
arguments to specify which model, framework, mode, platform, and docker
image to use, along with your path to the ImageNet TF Records that you
generated in step 3.

Substitute in your own `--data-location` and follow the steps in the
following example for throughput (using `--batch-size 64`):

```
$ python launch_benchmark.py \
	--platform fp32 \
	--model-name squeezenet \
	--mode inference \
	--framework tensorflow \
    --single-socket \
    --batch-size 64 \
	--docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
	--checkpoint ${checkpoint_dir} \
	--data-location /home/myuser/datasets/ImageNet_TFRecords \
	--verbose \
```

Or for latency (using `--batch-size 1`):

```
$ python launch_benchmark.py \
	--platform fp32 \
	--model-name squeezenet \
	--mode inference \
	--framework tensorflow \
    --single-socket \
    --batch-size 1 \
	--docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
	--checkpoint ${checkpoint_dir} \
	--data-location /home/myuser/datasets/ImageNet_TFRecords \
	--verbose \
```

5.  The log file is saved to:
models/benchmarks/common/tensorflow/logs/benchmark_squeezenet_inference.log

The tail of the log output when the benchmarking completes should look
something like this, when running for throughput with `--batch-size 64`:

```
step 246: 0.0758 sec
step 247: 0.0758 sec
step 248: 0.0756 sec
step 249: 0.0784 sec
step 250: 0.0759 sec
SqueezeNet Inference Summary:
            250 batches x 64 bs = total 16000 images evaluated
            batch size = 64
            throughput[med] = 841.0 image/sec
            latency[median] = 1.189 ms

lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
current path: /workspace/benchmarks
search path: /workspace/benchmarks/*/tensorflow/squeezenet/inference/fp32/model_init.py
Using model init: /workspace/benchmarks/image_recognition/tensorflow/squeezenet/inference/fp32/model_init.py
Received these standard args: Namespace(accuracy_only=False, batch_size=64, benchmark_only=False, checkpoint='/checkpoints', data_location='/dataset', framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='squeezenet', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=1, num_intra_threads=28, platform='fp32', single_socket=True, socket_id=0, verbose=True)
Received these custom args: []
taskset -c 0-27 python /workspace/intelai_models/image_recognition/tensorflow/squeezenet/fp32/train_squeezenet.py --data_location /dataset --batch_size 64 --num_inter_threads 1 --num_intra_threads 28 --model_dir /checkpoints --inference-only --verbose
PYTHONPATH: :/workspace/intelai_models
RUNCMD: python common/tensorflow/run_tf_benchmark.py         --framework=tensorflow         --model-name=squeezenet         --platform=fp32         --mode=inference         --model-source-dir=/workspace/models         --batch-size=64         --single-socket         --data-location=/dataset         --checkpoint=/checkpoints         --intelai-models=/workspace/intelai_models         --verbose
Batch Size: 64
Ran inference with batch size 64
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_squeezenet_inference.log
```

Or for latency (with `--batch-size 1`):

```
step 246: 0.0089 sec
step 247: 0.0088 sec
step 248: 0.0085 sec
step 249: 0.0095 sec
step 250: 0.0084 sec
SqueezeNet Inference Summary:
            250 batches x 1 bs = total 250 images evaluated
            batch size = 1
            throughput[med] = 119.1 image/sec
            latency[median] = 8.397 ms

lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
current path: /workspace/benchmarks
search path: /workspace/benchmarks/*/tensorflow/squeezenet/inference/fp32/model_init.py
Using model init: /workspace/benchmarks/image_recognition/tensorflow/squeezenet/inference/fp32/model_init.py
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=False, checkpoint='/checkpoints', data_location='/dataset', framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='squeezenet', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=1, num_intra_threads=28, platform='fp32', single_socket=True, socket_id=0, verbose=True)
Received these custom args: []
taskset -c 0-27 python /workspace/intelai_models/image_recognition/tensorflow/squeezenet/fp32/train_squeezenet.py --data_location /dataset --batch_size 1 --num_inter_threads 1 --num_intra_threads 28 --model_dir /checkpoints --inference-only --verbose
PYTHONPATH: :/workspace/intelai_models
RUNCMD: python common/tensorflow/run_tf_benchmark.py         --framework=tensorflow         --model-name=squeezenet         --platform=fp32         --mode=inference         --model-source-dir=/workspace/models         --batch-size=1         --single-socket         --data-location=/dataset         --checkpoint=/checkpoints         --intelai-models=/workspace/intelai_models         --verbose
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_squeezenet_inference.log
```
