# Inception V4

This document has instructions for how to run Inception V4 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference
other precisions are coming later.

## Int8 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
   repository:
   ```
   $ git clone https://github.com/IntelAI/models.git
   ```
   This repository includes launch scripts for running benchmarks.

2. A link to download the pre-trained model is coming soon.

3. If you would like to run Inception V4 inference and test for
   accuracy, you will need the ImageNet dataset. Benchmarking for latency
   and throughput do not require the ImageNet dataset.  Instructions for
   downloading the ImageNet dataset and converting it to the TF Records
   format and be found
   [here](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data).

4. Next, navigate to the `benchmarks` directory in your local clone of
   the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
   The `launch_benchmark.py` script in the `benchmarks` directory is
   used for starting a benchmarking run in a optimized TensorFlow docker
   container. It has arguments to specify which model, framework, mode,
   precision, and docker image to use, along with your path to the ImageNet
   TF Records that you generated in step 3.

   Inception V4 can be run to test accuracy or benchmarking throughput or
   latency. Use one of the following examples below, depending on your use
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
       --docker-image intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl  \
       --in-graph /home/$USER/inceptionv4_int8_pretrained_model.pb \
       --data-location /home/$USER/ImageNet_TFRecords
   ```

   For throughput benchmarking (using `--benchmark-only`, `--socket-id 0` and `--batch-size 240`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision int8 \
       --mode inference \
       --framework tensorflow \
       --benchmark-only \
       --batch-size 240 \
       --socket-id 0 \
       --docker-image intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl \
       --in-graph /home/$USER/inceptionv4_int8_pretrained_model.pb
   ```

   For latency (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision int8 \
       --mode inference \
       --framework tensorflow \
       --benchmark-only \
       --batch-size 1 \
       --socket-id 0 \
       --docker-image intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl \
       --in-graph /home/$USER/inceptionv4_int8_pretrained_model.pb
   ```

   The docker image (`intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl`)
   used in the commands above were built using
   [TensorFlow](git@github.com:tensorflow/tensorflow.git) master
   ([e889ea1](https://github.com/tensorflow/tensorflow/commit/e889ea1dd965c31c391106aa3518fc23d2689954)) and
   [PR #25765](https://github.com/tensorflow/tensorflow/pull/25765).

   Note that the `--verbose` flag can be added to any of the above commands
   to get additional debug output.

5. The log file is saved to the
   `intelai/models/benchmarks/common/tensorflow/logs` directory. Below are
   examples of what the tail of your log file should look like for the
   different configs.

   Example log tail when running for accuracy:
   ```
   ...
   Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7985, 0.9504)
   Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7983, 0.9504)
   Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7984, 0.9504)
   Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7984, 0.9504)
   lscpu_path_cmd = command -v lscpu
   lscpu located here: /usr/bin/lscpu
   Ran inference with batch size 100
   Log location outside container: <output directory>/benchmark_inceptionv4_inference_int8_20190306_221608.log
   ```

   Example log tail when benchmarking for throughput:
   ```
   [Running warmup steps...]
   steps = 10, 185.108768528 images/sec
   [Running benchmark steps...]
   steps = 10, 184.482999017 images/sec
   steps = 20, 184.561572444 images/sec
   steps = 30, 184.620504126 images/sec
   steps = 40, 183.900309054 images/sec
   steps = 50, 184.110358713 images/sec
   lscpu_path_cmd = command -v lscpu
   lscpu located here: /usr/bin/lscpu
   Ran inference with batch size 240
   Log location outside container: <output directory>/benchmark_inceptionv4_inference_int8_20190306_215858.log
   ```

   Example log tail when benchmarking for latency:
   ```
   [Running warmup steps...]
   steps = 10, 30.8738415788 images/sec
   [Running benchmark steps...]
   steps = 10, 31.8633787623 images/sec
   steps = 20, 31.1129375635 images/sec
   steps = 30, 31.2716048462 images/sec
   steps = 40, 31.9682931663 images/sec
   steps = 50, 31.6665962009 images/sec
   Latency: 31.936 ms
   lscpu_path_cmd = command -v lscpu
   lscpu located here: /usr/bin/lscpu
   Ran inference with batch size 1
   Log location outside container: <output directory>/benchmark_inceptionv4_inference_int8_20190306_215702.log
   ```

## FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
   repository:
   ```
   $ git clone https://github.com/IntelAI/models.git
   ```
   This repository includes launch scripts for running benchmarks.

2. A link to download the pre-trained model is coming soon.

3. If you would like to run Inception V4 inference and test for
   accuracy, you will need the ImageNet dataset. Benchmarking for latency
   and throughput do not require the ImageNet dataset.  Instructions for
   downloading the ImageNet dataset and converting it to the TF Records
   format and be found
   [here](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data).

4. Next, navigate to the `benchmarks` directory in your local clone of
   the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
   The `launch_benchmark.py` script in the `benchmarks` directory is
   used for starting a benchmarking run in a optimized TensorFlow docker
   container. It has arguments to specify which model, framework, mode,
   precision, and docker image to use, along with your path to the ImageNet
   TF Records that you generated in step 3.

   Inception V4 can be run to test accuracy or benchmarking throughput or
   latency. Use one of the following examples below, depending on your use
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
       --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
       --in-graph /home/$USER/inceptionv4_fp32_pretrained_model.pb \
       --data-location /home/$USER/ImageNet_TFRecords
   ```

   For throughput benchmarking (using `--benchmark-only`, `--socket-id 0` and `--batch-size 240`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision fp32 \
       --mode inference \
       --framework tensorflow \
       --benchmark-only \
       --batch-size 240 \
       --socket-id 0 \
       --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
       --in-graph /home/$USER/inceptionv4_fp32_pretrained_model.pb
   ```

   For latency (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):
   ```
   python launch_benchmark.py \
       --model-name inceptionv4 \
       --precision fp32 \
       --mode inference \
       --framework tensorflow \
       --benchmark-only \
       --batch-size 1 \
       --socket-id 0 \
       --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
       --in-graph /home/$USER/inceptionv4_fp32_pretrained_model.pb
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
   Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.8015, 0.9517)
   Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.8017, 0.9518)
   Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.8017, 0.9518)
   Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.8018, 0.9519)
   Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.8018, 0.9519)
   Ran inference with batch size 100
   Log location outside container: <output directory>/benchmark_inceptionv4_inference_fp32_20190308_182729.log
   ```

   Example log tail when benchmarking for throughput:
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

   Example log tail when benchmarking for latency:
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