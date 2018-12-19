# MobileNet V1

This document has instructions for how to run MobileNet V1 for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training is coming
later.

## FP32 Inference Instructions

1. Download the ImageNet dataset and convert it to the TF records format
   using the instructions
   [here](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data).

2. Download and extract the checkpoint files for the pretrained MobileNet
   V1 FP32 model:

   ```
   $ wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz

   $ tar -xvf mobilenet_v1_1.0_224.tgz
   x ./
   x ./mobilenet_v1_1.0_224.tflite
   x ./mobilenet_v1_1.0_224.ckpt.meta
   x ./mobilenet_v1_1.0_224.ckpt.index
   x ./mobilenet_v1_1.0_224.ckpt.data-00000-of-00001
   x ./mobilenet_v1_1.0_224_info.txt
   x ./mobilenet_v1_1.0_224_frozen.pb
   x ./mobilenet_v1_1.0_224_eval.pbtxt
   ```

3. Clone the [tensorflow/models](https://github.com/tensorflow/models)
   repository.

    ```
    $ git clone https://github.com/tensorflow/models
    ```

    The [tensorflow/models](https://github.com/tensorflow/models) files
    are used for dependencies when running benchmarking.

4. Clone the [intelai/models](https://github.com/IntelAI/models) repo
   and then navigate to the benchmarks directory:

   ```
   $ git clone git@github.com:IntelAI/models.git
   $ cd models/benchmarks
   ```

   Benchmarking can be run for either latency or throughput using the
   commands below.  The `--data-location` should be the path to the
   ImageNet validation data from step 1, the `--checkpoint` arg should
   be the path to the checkpoint files from step 2, and the
   `--model-source-dir` should point to the
   [tensorflow/models](https://github.com/tensorflow/models) repo that
   was cloned in step 3.

   * Run benchmarking for latency (with `--batch-size 1`):
     ```
     python launch_benchmark.py \
         --platform fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
         --model-source-dir /home/myuser/tensorflow/models  \
         --batch-size 1 \
         --single-socket \
         --data-location /dataset/Imagenet_Validation \
         --checkpoint /home/myuser/mobilenet_v1_fp32_pretrained_model
     ```
    * Run benchmarking for throughput (with `--batch-size 100`):
      ```
      python launch_benchmark.py \
         --platform fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
         --model-source-dir /home/myuser/tensorflow/models  \
         --batch-size 100 \
         --single-socket \
         --data-location /dataset/Imagenet_Validation \
         --checkpoint /home/myuser/mobilenet_v1_fp32_pretrained_model
      ```

5. The log files for each benchmarking run are saved at:
   `intelai/models/benchmarks/common/tensorflow/logs`.

   * Below is a sample log file snippet when benchmarking latency:
     ```
     2018-12-13 19:18:05.340448: step 80, 76.3 images/sec
     2018-12-13 19:18:05.459113: step 90, 82.6 images/sec
     2018-12-13 19:18:05.578525: step 100, 84.0 images/sec
     eval/Accuracy[0]
     eval/Recall_5[0.01]
     INFO:tensorflow:Finished evaluation at 2018-12-13-19:18:05
     self._total_images_per_sec = 830.8
     self._displayed_steps = 10
     Total images/sec = 83.1
     Latency ms/step = 12.0
     lscpu_path_cmd = command -v lscpu
     lscpu located here: /usr/bin/lscpu
     Ran inference with batch size 1
     Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_mobilenet_v1_inference_fp32_20181213_191800.log
     ```

   * Below is a sample log file snippet when benchmarking throughput:
     ```
     2018-12-14 00:56:39.793218: step 80, 183.7 images/sec
     2018-12-14 00:56:45.391790: step 90, 181.9 images/sec
     2018-12-14 00:56:51.082103: step 100, 178.3 images/sec
     eval/Accuracy[0.0012]
     eval/Recall_5[0.0051]
     INFO:tensorflow:Finished evaluation at 2018-12-14-00:56:51
     self._total_images_per_sec = 1795.6
     self._displayed_steps = 10
     Total images/sec = 179.6
     lscpu_path_cmd = command -v lscpu
     lscpu located here: /usr/bin/lscpu
     Ran inference with batch size 100
     Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_mobilenet_v1_inference_fp32_20181214_005550.log
     ```