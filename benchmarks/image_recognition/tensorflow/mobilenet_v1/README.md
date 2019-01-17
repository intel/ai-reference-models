# MobileNet V1

This document has instructions for how to run MobileNet V1 for the
following modes/precisions:
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
   $ git clone https://github.com/IntelAI/models.git
   $ cd models/benchmarks
   ```

   Benchmarking can be run for either latency or throughput using the
   commands below.  The `--data-location` should be the path to the
   ImageNet validation data from step 1, the `--checkpoint` arg should
   be the path to the checkpoint files from step 2, and the
   `--model-source-dir` should point to the
   [tensorflow/models](https://github.com/tensorflow/models) repo that
   was cloned in step 3.

   * Run benchmarking for latency (with `--batch-size 1` and `--checkpoint` with a path to the checkpoint file directory):
     ```
     python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
         --model-source-dir /home/myuser/tensorflow/models  \
         --batch-size 1 \
         --socket-id 0 \
         --data-location /dataset/Imagenet_Validation \
         --checkpoint /home/myuser/mobilenet_v1_fp32_pretrained_model
     ```
    * Run benchmarking for throughput (with `--batch-size 100` and `--checkpoint` with a path to the checkpoint file directory):
      ```
      python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
         --model-source-dir /home/myuser/tensorflow/models  \
         --batch-size 100 \
         --socket-id 0 \
         --data-location /dataset/Imagenet_Validation \
         --checkpoint /home/myuser/mobilenet_v1_fp32_pretrained_model
      ```
    * Run benchmarking for accuracy (with `--batch-size 100`, `--accuracy-only` and `--in-graph` with a path to the frozen graph .pb file):
      ```
      python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
         --model-source-dir /home/myuser/tensorflow/models  \
         --batch-size 100 \
         --accuracy-only \
         --data-location /dataset/Imagenet_Validation \
         --in-graph /home/myuser/mobilenet_v1_fp32_pretrained_model/mobilenet_v1_1.0_224_frozen.pb
      ```
      Note that the `--verbose` flag can be added to any of the above commands
      to get additional debug output.

5. The log files for each benchmarking run are saved at:
   `intelai/models/benchmarks/common/tensorflow/logs`.

   * Below is a sample log file snippet when benchmarking latency:
     ```
     2019-01-04 20:02:23.855441: step 80, 78.3 images/sec
     2019-01-04 20:02:23.974862: step 90, 83.7 images/sec
     2019-01-04 20:02:24.097476: step 100, 84.0 images/sec
     eval/Accuracy[0]
     eval/Recall_5[0]
     INFO:tensorflow:Finished evaluation at 2019-01-04-20:02:24
     self._total_images_per_sec = 809.6
     self._displayed_steps = 10
     Total images/sec = 81.0
     Latency ms/step = 12.4
     lscpu_path_cmd = command -v lscpu
     lscpu located here: /usr/bin/lscpu
     Ran inference with batch size 1
     Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_mobilenet_v1_inference_fp32_20190104_200218.log
     ```

   * Below is a sample log file snippet when benchmarking throughput:
     ```
     2019-01-04 20:06:01.151312: step 80, 184.0 images/sec
     2019-01-04 20:06:06.719081: step 90, 180.5 images/sec
     2019-01-04 20:06:12.346302: step 100, 174.1 images/sec
     eval/Accuracy[0.0009]
     eval/Recall_5[0.0049]
     INFO:tensorflow:Finished evaluation at 2019-01-04-20:06:12
     self._total_images_per_sec = 1810.2
     self._displayed_steps = 10
     Total images/sec = 181.0
     lscpu_path_cmd = command -v lscpu
     lscpu located here: /usr/bin/lscpu
     Ran inference with batch size 100
     Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_mobilenet_v1_inference_fp32_20190104_200512.log
     ```
   * Below is a sample lof file snippet when testing accuracy:
     ```
     Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7104, 0.8999)
     Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7103, 0.8999)
     Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7102, 0.8999)
     lscpu_path_cmd = command -v lscpu
     lscpu located here: /usr/bin/lscpu
     Ran inference with batch size 100
     Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_mobilenet_v1_inference_fp32_20190110_211648.log
     ```