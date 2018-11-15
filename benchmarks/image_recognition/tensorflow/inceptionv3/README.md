# inception v3

This document has instructions for how to run inception v3 for the
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

This repository includes launch scripts for running benchmarks and the
an optimized version of the inceptionv3 model code.

2. Clone the [tensorflow/models](https://github.com/tensorflow/models)
repository:

```
git clone git@github.com:tensorflow/models.git
```

This repository is used for dependencies that the inceptionv3 model
requires.

3. Download the pre-trained inceptionv3 model:

```
$ mkdir pretrained_models
$ pretrained_models
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/inceptionv3_int8_pretrained_model.pb
```

4. Build a docker image using the quantized TensorFlow
[branch](https://github.com/tensorflow/tensorflow/pull/21483)
in the official TensorFlow repository with `--config=mkl`. More instructions on
[how to build from source](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#inpage-nav-5).

5. If you would like to run inceptionv3 inference and test for
accurancy, you will need the ImageNet dataset.  Benchmarking for latency
and throughput do not require the ImageNet dataset.

Register and download the
[ImageNet dataset](http://image-net.org/download-images).

Once you have the raw ImageNet dataset downloaded, we need to convert
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

6. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a TensorFlow docker container.
It has arguments to specify which model, framework, mode, platform, and
docker image to use, along with your path to the ImageNet TF Records
that you generated in step 5.

Substitute in your own `--data-location` (from step 5, for accuracy
only), `--in-graph` pretrained model file path (from step 3),
`--model-source-dir` for the location where you cloned the
[tensorflow/models](https://github.com/tensorflow/models) repo
(from step 2), and the name/tag for your docker image (from step 4).

Inceptionv3 can be run for accuracy, latency benchmarking, or throughput
benchmarking.  Use one of the following examples below, depending on
your use case.

For accuracy (using your `--data-location`, `--accuracy-only` and
`--batch-size 128`):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 128 \
	--docker-image tf_int8_docker_image \
	--in-graph /home/myuser/pretrained_models/inceptionv3_int8_pretrained_model.pb \
	--data-location /home/myuser/datasets/ImageNet_TFRecords \
	--verbose \
	-- input_height=299 input_width=299
```

For latency (using `--benchmark-only`, `--single-socket` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --single-socket \
	--docker-image tf_int8_docker_image \
	--in-graph /home/myuser/pretrained_models/inceptionv3_int8_pretrained_model.pb \
	--data-location /home/myuser/datasets/ImageNet_TFRecords \
	--verbose \
	-- input_height=299 input_width=299
```

For throughput (using `--benchmark-only`, `--single-socket` and `--batch-size 128`):

```
python launch_benchmark.py \
    --model-name inceptionv3 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 128 \
    --single-socket \
	--docker-image tf_int8_docker_image \
	--in-graph /home/myuser/pretrained_models/inceptionv3_int8_pretrained_model.pb \
	--data-location /home/myuser/datasets/ImageNet_TFRecords \
	--verbose \
	-- input_height=299 input_width=299
```

7.  The log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory.  Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for accuracy:

```
Processed 49920 images. (Top1 accuracy, Top5 accuracy) = (0.7667, 0.9336)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
current path: /workspace/benchmarks
search path: /workspace/benchmarks/*/tensorflow/inceptionv3/inference/int8/model_init.py
Using model init: /workspace/benchmarks/image_recognition/tensorflow/inceptionv3/inference/int8/model_init.py
Received these standard args: Namespace(accuracy_only=True, batch_size=128, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/final_int8_inceptionv3.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='inceptionv3', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=False, socket_id=0, verbose=True)
Received these custom args: ['--input-height=299', '--input-width=299']
Command: python /workspace/intelai_models/image_recognition/tensorflow/inceptionv3/int8/accuracy.py --input_height=299 --input_width=299 --num_intra_threads=56 --num_inter_threads=2 --batch_size=128 --input_graph=/in_graph/final_int8_inceptionv3.pb --data_location=/dataset
PYTHONPATH: :/workspace/intelai_models:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --model-name=inceptionv3 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=128 --data-location=/dataset  --verbose --accuracy-only  --in-graph=/in_graph/final_int8_inceptionv3.pb --input-height=299 --input-width=299
Batch Size: 128
Ran inference with batch size 128
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_inceptionv3_inference.log
```

Example log tail when benchmarking for latency:
```
[Running warmup steps...]
steps = 10, 56.7603220786 images/sec
[Running benchmark steps...]
steps = 10, 56.4334593598 images/sec
steps = 20, 56.9568712656 images/sec
steps = 30, 57.126762098 images/sec
steps = 40, 56.7633947301 images/sec
steps = 50, 56.7342179659 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
current path: /workspace/benchmarks
search path: /workspace/benchmarks/*/tensorflow/inceptionv3/inference/int8/model_init.py
Using model init: /workspace/benchmarks/image_recognition/tensorflow/inceptionv3/inference/int8/model_init.py
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=True, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/final_int8_inceptionv3.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='inceptionv3', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=True, socket_id=0, verbose=True)
Received these custom args: ['--input-height=299', '--input-width=299']
Command: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/image_recognition/tensorflow/inceptionv3/int8/benchmark.py --input_height=299 --input_width=299 --warmup_steps=10 --num_intra_threads=56 --num_inter_threads=2 --batch_size=1 --input_graph=/in_graph/final_int8_inceptionv3.pb --steps=50
PYTHONPATH: :/workspace/intelai_models:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --model-name=inceptionv3 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=1 --data-location=/dataset --single-socket --verbose  --benchmark-only --in-graph=/in_graph/final_int8_inceptionv3.pb --input-height=299 --input-width=299
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_inceptionv3_inference.log
```

Example log tail when benchmarking for throughput:
```
[Running warmup steps...]
steps = 10, 334.715689471 images/sec
[Running benchmark steps...]
steps = 10, 334.040719213 images/sec
steps = 20, 336.002964046 images/sec
steps = 30, 335.952712516 images/sec
steps = 40, 333.1258265 images/sec
steps = 50, 335.220831897 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
current path: /workspace/benchmarks
search path: /workspace/benchmarks/*/tensorflow/inceptionv3/inference/int8/model_init.py
Using model init: /workspace/benchmarks/image_recognition/tensorflow/inceptionv3/inference/int8/model_init.py
Received these standard args: Namespace(accuracy_only=False, batch_size=128, benchmark_only=True, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/final_int8_inceptionv3.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='inceptionv3', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=True, socket_id=0, verbose=True)
Received these custom args: ['--input-height=299', '--input-width=299']
Command: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/image_recognition/tensorflow/inceptionv3/int8/benchmark.py --input_height=299 --input_width=299 --warmup_steps=10 --num_intra_threads=56 --num_inter_threads=2 --batch_size=128 --input_graph=/in_graph/final_int8_inceptionv3.pb --steps=50
PYTHONPATH: :/workspace/intelai_models:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --model-name=inceptionv3 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=128 --data-location=/dataset --single-socket --verbose  --benchmark-only --in-graph=/in_graph/final_int8_inceptionv3.pb --input-height=299 --input-width=299
Batch Size: 128
Ran inference with batch size 128
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_inceptionv3_inference_int8.log
```