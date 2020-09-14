# Inception V4

This document has instructions for how to run Inception V4 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model training and inference for
other precisions are coming later.

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
   This repository includes launch scripts for running the model.

2. Download the pretrained model:
   ```
   $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv4_int8_pretrained_model.pb
   ```

3. If you would like to run Inception V4 inference and test for
   accuracy, you will need the ImageNet dataset.  It is not necessary for batch or online inference, you have the option of using synthetic data instead.

   Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
   After running the conversion script you should have a directory with the
   ImageNet dataset in the TF records format.

4. Next, navigate to the `benchmarks` directory in your local clone of
   the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
   The `launch_benchmark.py` script in the `benchmarks` directory is
   used for starting a model run in a optimized TensorFlow docker
   container. It has arguments to specify which model, framework, mode,
   precision, and docker image to use, along with your path to the ImageNet
   TF Records that you generated in step 3.

   Inception V4 can be run to test accuracy, batch inference, or
   online inference. Use one of the following examples below, depending on your use
   case.

   For accuracy (using your `--data-location`, `--accuracy-only` and
   `--batch-size 100`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision int8 \
       --mode inference \
       --framework tensorflow \
       --accuracy-only \
       --batch-size 100 \
       --socket-id 0 \
       --docker-image intel/intel-optimized-tensorflow:2.3.0 \
       --in-graph /home/<user>/inceptionv4_int8_pretrained_model.pb \
       --data-location /home/<user>/ImageNet_TFRecords
   ```

   For batch inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 240`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision int8 \
       --mode inference \
       --framework tensorflow \
       --benchmark-only \
       --batch-size 240 \
       --socket-id 0 \
       --docker-image intel/intel-optimized-tensorflow:2.3.0 \
       --in-graph /home/<user>/inceptionv4_int8_pretrained_model.pb
   ```

   For online inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision int8 \
       --mode inference \
       --framework tensorflow \
       --benchmark-only \
       --batch-size 1 \
       --socket-id 0 \
       --docker-image intel/intel-optimized-tensorflow:2.3.0 \
       --in-graph /home/<user>/inceptionv4_int8_pretrained_model.pb
   ```

   Note that the `--verbose` flag can be added to any of the above commands
   to get additional debug output.

5. The log file is saved to the
   `intelai/models/benchmarks/common/tensorflow/logs` directory. Below are
   examples of what the tail of your log file should look like for the
   different configs.

   Example log tail when running for accuracy:
   ```
   ...
   Iteration time: 685.1976 ms
   Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7985, 0.9504)
   Iteration time: 686.3845 ms
   Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7983, 0.9504)
   Iteration time: 686.7021 ms
   Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7984, 0.9504)
   Iteration time: 685.8914 ms
   Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7984, 0.9504)
   Ran inference with batch size 100
   Log location outside container: <output directory>/benchmark_inceptionv4_inference_int8_20190306_221608.log
   ```

   Example log tail when running for batch inference:
   ```
    [Running warmup steps...]
    steps = 10, 184.497605972 images/sec
    [Running benchmark steps...]
    steps = 10, 184.664702184 images/sec
    steps = 20, 184.938455688 images/sec
    steps = 30, 184.454197634 images/sec
    steps = 40, 184.491891402 images/sec
    steps = 50, 184.390001575 images/sec
    Ran inference with batch size 240
    Log location outside container: <output directory>/benchmark_inceptionv4_inference_int8_20190415_233517.log
   ```

   Example log tail when running for online inference:
   ```
    [Running warmup steps...]
    steps = 10, 32.6095380262 images/sec
    [Running benchmark steps...]
    steps = 10, 32.9024373024 images/sec
    steps = 20, 32.5328989723 images/sec
    steps = 30, 32.5988932413 images/sec
    steps = 40, 31.3991914957 images/sec
    steps = 50, 32.7053998207 images/sec
    Latency: 30.598 ms
    Ran inference with batch size 1
    Log location outside container: <output directory>/benchmark_inceptionv4_inference_int8_20190415_232441.log
   ```

## FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
   repository:
   ```
   $ git clone https://github.com/IntelAI/models.git
   ```
   This repository includes launch scripts for running the model.

2. Download the pretrained model:
   ```
   $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv4_fp32_pretrained_model.pb
   ```

3. If you would like to run Inception V4 inference and test for
   accuracy, you will need the ImageNet dataset. Running for online
   and batch inference do not require the ImageNet dataset.

   Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
   After running the conversion script you should have a directory with the
   ImageNet dataset in the TF records format.

4. Next, navigate to the `benchmarks` directory in your local clone of
   the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
   The `launch_benchmark.py` script in the `benchmarks` directory is
   used for starting a model run in a optimized TensorFlow docker
   container. It has arguments to specify which model, framework, mode,
   precision, and docker image to use, along with your path to the ImageNet
   TF Records that you generated in step 3.

   Inception V4 can be run to test accuracy, batch inference, or
   online inference. Use one of the following examples below, depending on your use
   case.

   For accuracy (using your `--data-location`, `--accuracy-only` and
   `--batch-size 100`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision fp32 \
       --mode inference \
       --framework tensorflow \
       --accuracy-only \
       --batch-size 100 \
       --socket-id 0 \
       --docker-image intel/intel-optimized-tensorflow:2.3.0 \
       --in-graph /home/<user>/inceptionv4_fp32_pretrained_model.pb \
       --data-location /home/<user>/ImageNet_TFRecords
   ```

   For batch inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 240`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision fp32 \
       --mode inference \
       --framework tensorflow \
       --benchmark-only \
       --batch-size 240 \
       --socket-id 0 \
       --docker-image intel/intel-optimized-tensorflow:2.3.0 \
       --in-graph /home/<user>/inceptionv4_fp32_pretrained_model.pb
   ```

   For online inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision fp32 \
       --mode inference \
       --framework tensorflow \
       --benchmark-only \
       --batch-size 1 \
       --socket-id 0 \
       --docker-image intel/intel-optimized-tensorflow:2.3.0 \
       --in-graph /home/<user>/inceptionv4_fp32_pretrained_model.pb
   ```

   Note that the `--verbose` flag can be added to any of the above commands
   to get additional debug output.

5. The log file is saved to the
   `intelai/models/benchmarks/common/tensorflow/logs` directory. Below are
   examples of what the tail of your log file should look like for the
   different configs.

   Example log tail when running for accuracy:
   ```
   ...
   Iteration time: 1337.8728 ms
   Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.8015, 0.9517)
   Iteration time: 1331.8253 ms
   Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.8017, 0.9518)
   Iteration time: 1339.1553 ms
   Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.8017, 0.9518)
   Iteration time: 1334.5991 ms
   Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.8018, 0.9519)
   Iteration time: 1336.1905 ms
   Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.8018, 0.9519)
   Ran inference with batch size 100
   Log location outside container: <output directory>/benchmark_inceptionv4_inference_fp32_20190308_182729.log
   ```

   Example log tail when running for batch inference:
   ```
   [Running warmup steps...]
   steps = 10, 91.4372832625 images/sec
   [Running benchmark steps...]
   steps = 10, 91.0217283977 images/sec
   steps = 20, 90.8331507586 images/sec
   steps = 30, 91.1284943026 images/sec
   steps = 40, 91.1885998597 images/sec
   steps = 50, 91.1905741783 images/sec
   Ran inference with batch size 240
   Log location outside container: <output directory>/benchmark_inceptionv4_inference_fp32_20190308_184431.log
   ```

   Example log tail when running for online inference:
   ```
   [Running warmup steps...]
   steps = 10, 15.6993019295 images/sec
   [Running benchmark steps...]
   steps = 10, 16.3553780883 images/sec
   steps = 20, 15.771143231 images/sec
   steps = 30, 15.7133587586 images/sec
   steps = 40, 16.0477494988 images/sec
   steps = 50, 15.483992912 images/sec
   Latency: 63.534 ms
   Ran inference with batch size 1
   Log location outside container: <output directory>/benchmark_inceptionv4_inference_fp32_20190307_221954.log
   ```
