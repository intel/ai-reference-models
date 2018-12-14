# ResNet101

This document has instructions for how to run ResNet101 for the
following modes/platforms:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

## Int8 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone git@github.com:IntelAI/models.git
```

This repository includes launch scripts for running benchmarks and the
an optimized version of the ResNet101 model code.

2. Download the pre-trained ResNet101 model:

```
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/resnet101_int8_pretrained_model.pb
```

3. Build a docker image using master of the official
[TensorFlow](https://github.com/tensorflow/tensorflow) repository with
`--config=mkl`. More instructions on
[how to build from source](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#inpage-nav-5).

4. If you would like to run ResNet101 inference and test for
accurancy, you will need the full ImageNet dataset.

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
TF Records that you generated in step 4.

Substitute in your own `--data-location` (from step 4, for accuracy
only), `--in-graph` pre-trained model file path (from step 2),
and the name/tag for your docker image (from step 3).

ResNet101 can be run for accuracy or performance benchmarking. Use one of
the following examples below, depending on your use case.

For accuracy (using your `--data-location`,`--in-graph`, `--accuracy-only` and
`--batch-size 100`):

```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --model-name resnet101 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --docker-image tf_int8_docker_image \
    --data-location /home/myuser/dataset/FullImageNetData_directory \
    --in-graph=/home/myuser/resnet101_int8_pretrained_model.pb
```

For latency (using `--benchmark-only`, `--single-socket` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name resnet101 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --single-socket \
    --docker-image tf_int8_docker_image \
    --in-graph=/home/myuser/inceptionv3_int8_pretrained_model.pb
```

For throughput (using `--benchmark-only`, `--single-socket` and `--batch-size 128`):

```
python launch_benchmark.py \
    --model-name resnet101 \
    --platform int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 128 \
    --single-socket \
    --docker-image tf_int8_docker_image \
    --in-graph=/home/myuser/inceptionv3_int8_pretrained_model.pb
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

6. The log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for accuracy:
```
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7696, 0.9309)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=True, batch_size=100, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/resnet101_int8_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet101', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='int8', single_socket=False, socket_id=0, use_case='image_recognition', verbose=True)
Received these custom args: []
PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet101 --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=100 --data-location=/dataset  --verbose --in-graph=/in_graph/resnet101_int8_pretrained_model.pb --accuracy-only
Batch Size: 100
Ran inference with batch size 100
Log location outside container: /home/myuser/intel-models/benchmarks/common/tensorflow/logs/benchmark_resnet101_inference.log
```

Example log tail when benchmarking for latency:
```
[Running warmup steps...]
steps = 10, 53.3022912987 images/sec
steps = 20, 54.8999856019 images/sec
steps = 30, 54.5288420286 images/sec
steps = 40, 54.3775556506 images/sec
[Running benchmark steps...]
steps = 10, 537.185143822 images/sec
steps = 20, 268.75073286 images/sec
steps = 30, 179.033434653 images/sec
steps = 40, 134.356211634 images/sec
steps = 50, 107.403547389 images/sec
steps = 60, 89.3812766404 images/sec
steps = 70, 76.565932747 images/sec
steps = 80, 67.0330362294 images/sec
steps = 90, 59.6184242546 images/sec
steps = 100, 53.6588898046 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: /home/myuser/intel-models/benchmarks/common/tensorflow/logs/benchmark_resnet101_inference.log
```

Example log tail when benchmarking for throughput:
```
[Running warmup steps...]
steps = 10, 261.899492271 images/sec
steps = 20, 264.73629574 images/sec
steps = 30, 264.42726748 images/sec
steps = 40, 266.918027016 images/sec
[Running benchmark steps...]
steps = 10, 2649.86314633 images/sec
steps = 20, 1321.40108889 images/sec
steps = 30, 880.866111729 images/sec
steps = 40, 660.864211524 images/sec
steps = 50, 528.715408097 images/sec
steps = 60, 440.655396861 images/sec
steps = 70, 377.691701833 images/sec
steps = 80, 330.564774873 images/sec
steps = 90, 293.794278281 images/sec
steps = 100, 264.372863325 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 128
Log location outside container: /home/myuser/intel-models/benchmarks/common/tensorflow/logs/benchmark_resnet101_inference.log
```

## FP32 Inference Instructions

1. Clone the
[intelai/models](https://github.com/intelai/models)
repository
    ```
    $ git clone git@github.com:IntelAI/models.git
    ```

2. Download the pre-trained ResNet101 model:

    ```
    $ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/resnet101_fp32_pretrained_model.pb
    ```
3. Download ImageNet dataset.

    This step is required only required for running accuracy, for running benchmark we do not need to provide dataset.

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
4. Run the benchmark.

    For latency measurements set `--batch-size 1` and for throughput benchmarking set `--batch-size 128`

    ```
    $ cd /home/myuser/models/benchmarks
    $ python launch_benchmark.py
        --framework tensorflow
        --platform fp32
        --mode inference
        --model-name resnet101
        --batch-size 128
        --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
        --in-graph /home/myuser/trained_models/resnet101_fp32_pretrained_model.pb
        --single-socket
        --verbose
    ```

    The log file is saved to: `models/benchmarks/common/tensorflow/logs/`.

    The tail of the log output when the benchmarking completes should look something like this:

      ```
        steps = 10, 1342.80359717 images/sec
        steps = 20, 670.767434629 images/sec
        steps = 30, 446.319515464 images/sec
        steps = 40, 334.314206698 images/sec
        steps = 50, 267.251707323 images/sec
        steps = 60, 222.571395923 images/sec
        steps = 70, 190.724044039 images/sec
        steps = 80, 166.881224428 images/sec
        steps = 90, 148.365949039 images/sec
        steps = 100, 133.594262281 images/sec
        lscpu_path_cmd = command -v lscpu
        lscpu located here: /usr/bin/lscpu
        Received these standard args: Namespace(accuracy_only=False, batch_size=128, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/resnet101_fp32_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet101', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
        Received these custom args: []
        Current directory: /workspace/benchmarks
        Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/fp32/benchmark.py --batch_size=128 --num_inter_threads=2 --input_graph=/in_graph/resnet101_fp32_pretrained_model.pb --num_intra_threads=56
        PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
        RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet101 --platform=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=128 --single-socket --verbose --in-graph=/in_graph/resnet101_fp32_pretrained_model.pb       --data-location=/dataset
        Batch Size: 128
        Ran inference with batch size 128
        Log location outside container: /home/myuser/resnet101/intel-models/benchmarks/common/tensorflow/logs/benchmark_resnet101_inference_fp32_20181205_194744.log
      ```

5. Run for accuracy
    ```
    $ cd /home/myuser/models/benchmarks
    $ python launch_benchmark.py
        --framework tensorflow
        --platform fp32
        --mode inference
        --model-name resnet101
        --batch-size 100
        --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
        --in-graph /home/myuser/trained_models/resnet101_fp32_pretrained_model.pb
        --data-location /home/myuser/imagenet_validation_dataset
        --accuracy-only
        --single-socket
        --verbose
    ```

    The log file is saved to: `/home/myuser/resnet101/intel-models/benchmarks/common/tensorflow/logs/`.

    The tail of the log output when the benchmarking completes should look something like this:

      ```
        Processed 49300 images. (Top1 accuracy, Top5 accuracy) = (0.7641, 0.9290)
        Processed 49400 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9289)
        Processed 49500 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9289)
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7638, 0.9289)
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7638, 0.9288)
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7640, 0.9289)
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7640, 0.9289)
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7640, 0.9289)
        lscpu_path_cmd = command -v lscpu
        lscpu located here: /usr/bin/lscpu
        Received these standard args: Namespace(accuracy_only=True, batch_size=100, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/resnet101_fp32_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet101', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
        Received these custom args: []
        Current directory: /workspace/benchmarks
        Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/fp32/accuracy.py --batch_size=100 --num_inter_threads=2 --input_graph=/in_graph/resnet101_fp32_pretrained_model.pb --num_intra_threads=56 --data_location=/dataset
        PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
        RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet101 --platform=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=100 --single-socket --accuracy-only  --verbose --in-graph=/in_graph/resnet101_fp32_pretrained_model.pb --data-location=/dataset
        Batch Size: 100
        Ran inference with batch size 100
        Log location outside container: /home/myuser/resnet101/intel-models/benchmarks/common/tensorflow/logs/benchmark_resnet101_inference_fp32_20181207_221503.log
    ```

