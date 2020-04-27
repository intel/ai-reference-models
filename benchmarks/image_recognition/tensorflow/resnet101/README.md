# ResNet101

This document has instructions for how to run ResNet101 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

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

This repository includes launch scripts for running 
an optimized version of the ResNet101 model code.

2. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet101_int8_pretrained_model.pb
```

3. If you would like to run ResNet101 inference with real data or test for
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
$ ll /home/<user>/datasets/ImageNet_TFRecords
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
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the ImageNet
TF Records that you generated in step 3.

Substitute in your own `--data-location` (from step 3, for accuracy
only) and `--in-graph` pre-trained model file path (from step 2).

ResNet101 can be run for testing accuracy or performance. Use one of
the following examples below, depending on your use case.

For accuracy (using your `--data-location`,`--in-graph`, `--accuracy-only` and
`--batch-size 100`):

```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --model-name resnet101 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --data-location /home/<user>/dataset/FullImageNetData_directory \
    --in-graph=/home/<user>/resnet101_int8_pretrained_model.pb
```

When running for performance, it is optional to specify the
number of `warmup_steps` and `steps` as extra args, as shown in the
commands below. If these values are not specified, the script will
default to use `warmup_steps=40` and `steps=100`.

For online inference with dummy data (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name resnet101 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --in-graph=/home/<user>/resnet101_int8_pretrained_model.pb \
    -- warmup_steps=50 steps=500
```

For online inference with ImageNet data (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name resnet101 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --socket-id 0 \
    --data-location /home/<user>/dataset/FullImageNetData_directory \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --in-graph=/home/<user>/resnet101_int8_pretrained_model.pb \
    -- warmup_steps=50 steps=500
```

For batch inference with dummy data (using `--benchmark-only`, `--socket-id 0` and `--batch-size 128`):

```
python launch_benchmark.py \
    --model-name resnet101 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 128 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --in-graph=/home/<user>/resnet101_int8_pretrained_model.pb \
    -- warmup_steps=50 steps=500
```

For batch inference with ImageNet data (using `--benchmark-only`, `--socket-id 0` and `--batch-size 128`):

```
python launch_benchmark.py \
    --model-name resnet101 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 128 \
    --data-location /home/<user>/dataset/FullImageNetData_directory \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --in-graph=/home/<user>/resnet101_int8_pretrained_model.pb \
    -- warmup_steps=50 steps=500
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..

6. The log file is saved to the value
of `--output-dir`. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for accuracy:
```
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7690, 0.9304)
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7691, 0.9305)
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7691, 0.9305)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190104_205838.log
```

Example log tail when running for online inference:
```
...
steps = 470, 48.3195530058 images/sec
steps = 480, 47.2792312364 images/sec
steps = 490, 46.3175214744 images/sec
steps = 500, 45.4044245083 images/sec
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190223_191406.log
```

Example log tail when running for batch inference:
```
...
steps = 470, 328.906266308 images/sec
steps = 480, 322.0451309 images/sec
steps = 490, 315.455582114 images/sec
steps = 500, 309.142758646 images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190223_192438.log
```

## FP32 Inference Instructions

1. Clone the
[intelai/models](https://github.com/intelai/models)
repository
    ```
    $ git clone https://github.com/IntelAI/models.git
    ```

2. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet101_fp32_pretrained_model.pb
```

3. Download ImageNet dataset.

    This step is only required for running accuracy, for running online and batch inference we do not need to provide dataset.

    Register and download the ImageNet dataset. Once you have the raw ImageNet dataset downloaded, we need to convert
    it to the TFRecord format. The TensorFlow models repo provides
    [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
    to download, process and convert the ImageNet dataset to the TF records format. After converting data, you should have a directory
    with the sharded dataset something like below, we only need `validation-*` files, discard `train-*` files:
    ```
    $ ll /home/<user>/datasets/ImageNet_TFRecords
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
4. Run the script.

    For online inference measurements with dummy data set `--batch-size 1` and for batch inference set `--batch-size 128`

    ```
    $ cd /home/<user>/models/benchmarks
    $ python launch_benchmark.py \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --model-name resnet101 \
        --batch-size 128 \
        --docker-image intel/intel-optimized-tensorflow:2.1.0 \
        --in-graph /home/<user>/trained_models/resnet101_fp32_pretrained_model.pb \
        --socket-id 0
    ```

    The log file is saved to the value of `--output-dir`.

    The tail of the log output when the run completes should look something like this:

    ```
    steps = 70, 193.428695737 images/sec
    steps = 80, 169.258177508 images/sec
    steps = 90, 150.457869027 images/sec
    steps = 100, 135.433960175 images/sec
    Ran inference with batch size 128
    Log location outside container: {--output-dir value}/benchmark_resnet101_inference_fp32_20190104_204615.log
    ```

5. Run for accuracy
    ```
    $ cd /home/<user>/models/benchmarks
    $ python launch_benchmark.py \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --model-name resnet101 \
        --batch-size 100 \
        --docker-image intel/intel-optimized-tensorflow:2.1.0 \
        --in-graph /home/<user>/trained_models/resnet101_fp32_pretrained_model.pb \
        --data-location /home/<user>/imagenet_validation_dataset \
        --accuracy-only \
        --socket-id 0
    ```

    The log file is saved to the value of `--output-dir`.

    The tail of the log output when the run completes should look something like this:

    ```
    Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9289)
    Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9290)
    Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9289)
    Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7641, 0.9289)
    Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7640, 0.9289)
    Ran inference with batch size 100
    Log location outside container: {--output-dir value}/benchmark_resnet101_inference_fp32_20190104_201506.log
    ```

   Note that the `--verbose` or `--output-dir` flag can be added to any of the above
   commands to get additional debug output or change the default output location.
