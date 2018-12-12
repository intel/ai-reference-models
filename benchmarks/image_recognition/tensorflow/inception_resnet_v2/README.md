# Inception ResNet V2

This document has instructions for how to run Inception ResNet V2 for the
following modes/platforms:
* [Int8 inference](#int8-inference-instructions)

Benchmarking instructions and scripts for Inception ResNet V2 model inference on 'Int8'
platforms.

## Int8 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone git@github.com:IntelAI/models.git
```

This repository includes launch scripts for running benchmarks and the
an optimized version of the Inception ResNet V2 model code.

2. Download the pre-trained Inception ResNet V2 model:

```
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/inception_resnet_v2_int8_pretrained_model.pb
```

3. Build a docker image using master of the official
[TensorFlow](https://github.com/tensorflow/tensorflow) repository with
`--config=mkl`. More instructions on
[how to build from source](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#inpage-nav-5).

4. If you would like to run Inception ResNet V2 inference and test for
accurancy, you will need the full ImageNet dataset. Benchmarking for latency
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

5. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
platform, and docker image to use, along with your path to the ImageNet
TF Records that you generated in step 5.

Substitute in your own `--data-location` (from step 5, for accuracy
only), `--in-graph` pre-trained model file path (from step 3),
`--model-source-dir` for the location where you cloned the
[tensorflow/models](https://github.com/tensorflow/models) repo
(from step 2), and the name/tag for your docker image (from step 4).

Inception ResNet V2 can be run for accuracy, latency benchmarking, or throughput
benchmarking. Use one of the following examples below, depending on
your use case.

For accuracy (using your `--data-location`, `--accuracy-only` and
`--batch-size 100`):

```
python launch_benchmark.py \
    --model-name inception_resnet_v2 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/final_int8_inception_resnet_v2_graph.pb \
    --data-location /home/myuser/datasets/ImageNet_TFRecords \
```

For latency (using `--benchmark-only`, `--single-socket` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name inception_resnet_v2 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --single-socket \
    --docker-image tf_int8_docker_image \
    --in-graph /home/myuser/inceptionv3_int8_pretrained_model.pb \
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
    --in-graph /home/myuser/inceptionv3_int8_pretrained_model.pb \
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

6. The log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for accuracy:

```
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.8018, 0.9519)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=True, batch_size=100, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/inception_resnet_v2_int8_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='inception_resnet_v2', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=False, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
Current directory: /workspace/benchmarks
Running: python /workspace/intelai_models/int8/eval_image_classifier_accuracy.py --input_graph=/in_graph/inception_resnet_v2_int8_pretrained_model.pb --data_location=/dataset --input_height=299 --input_width=299 --num_inter_threads=2 --num_intra_threads=56 --output_layer=InceptionResnetV2/Logits/Predictions --batch_size=100
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=inception_resnet_v2 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=100  --accuracy-only  --verbose --in-graph=/in_graph/inception_resnet_v2_int8_pretrained_model.pb --data-location=/dataset
Batch Size: 100
Ran inference with batch size 100
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_inception_resnet_v2_inference_int8_20181212_204045.log
```

Example log tail when benchmarking for latency:
```
Iteration 39: 0.051 sec
Iteration 40: 0.052 sec
Average time: 0.051 sec
Batch size = 1
Latency: 51.476 ms
Throughput: 19.427 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=True, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/inception_resnet_v2_int8_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='inception_resnet_v2', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=1, num_intra_threads=28, platform='int8', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
Current directory: /workspace/benchmarks
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/int8/eval_image_classifier_benchmark.py --input-graph=/in_graph/inception_resnet_v2_int8_pretrained_model.pb --inter-op-parallelism-threads=1 --intra-op-parallelism-threads=28 --batch-size=1
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=inception_resnet_v2 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=1 --single-socket  --benchmark-only --verbose --in-graph=/in_graph/inception_resnet_v2_int8_pretrained_model.pb --data-location=/dataset
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_inception_resnet_v2_inference_int8_20181212_213939.log
```

Example log tail when benchmarking for throughput:
```
Iteration 39: 0.976 sec
Iteration 40: 0.977 sec
Average time: 0.976 sec
Batch size = 128
Throughput: 131.141 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=128, benchmark_only=True, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/inception_resnet_v2_int8_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='inception_resnet_v2', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=1, num_intra_threads=28, platform='int8', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
Current directory: /workspace/benchmarks
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/int8/eval_image_classifier_benchmark.py --input-graph=/in_graph/inception_resnet_v2_int8_pretrained_model.pb --inter-op-parallelism-threads=1 --intra-op-parallelism-threads=28 --batch-size=128
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=inception_resnet_v2 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=128 --single-socket  --benchmark-only --verbose --in-graph=/in_graph/inception_resnet_v2_int8_pretrained_model.pb --data-location=/dataset
Batch Size: 128
Ran inference with batch size 128
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_inception_resnet_v2_inference_int8_20181212_214327.log
```