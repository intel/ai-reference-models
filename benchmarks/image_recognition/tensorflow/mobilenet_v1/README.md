# MobileNet V1

This document has instructions for how to run MobileNet V1 for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training is coming
later.


## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Download ImageNet dataset.

    This step is required only for running accuracy, for running benchmark we do not need to provide dataset.

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
2. A link to download the pre-trained model is coming soon.

3. Clone the [intelai/models](https://github.com/intelai/models) repo
    and then run the benchmarking scripts for either benchmarking throughput,
    latency or accuracy. For --dataset-location in accuracy run, please use the ImageNet validation data path from step 1.
    Each benchmark run has user configurable arguments separated from regular arguments by '--' at the end of the command.
    Unless configured, these arguments will run with default values. Below are the example codes for each benchmark case:

    ```
    $ git clone https://github.com/IntelAI/models.git

    $ cd benchmarks
    ```

    For throughput (using `--benchmark-only`, `--socket-id 0` and `--batch-size 240`):
    ```
    python launch_benchmark.py  \
         --model-name mobilenet_v1 \
         --precision int8 \
         --mode inference \
         --framework tensorflow \
         --benchmark-only \
         --batch-size 240  \
         --socket-id 0 \
         --in-graph /home/<user>/mobilenetv1_int8_pretrained_model.pb  \
         --docker-image intelaipg/intel-optimized-tensorflow:1.14 \
         -- input_height=224 input_width=224 warmup_steps=10 steps=50 \
         input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
    ```

    For latency (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`)
    ```
    python launch_benchmark.py  \
         --model-name mobilenet_v1 \
         --precision int8 \
         --mode inference \
         --framework tensorflow \
         --benchmark-only \
         --batch-size 1  \
         --socket-id 0 \
         --in-graph /home/<user>/mobilenetv1_int8_pretrained_model.pb  \
         --docker-image intelaipg/intel-optimized-tensorflow:1.14 \
         -- input_height=224 input_width=224 warmup_steps=10 steps=50 \
         input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
    ```

    For accuracy (using your `--data-location`, `--accuracy-only` and
    `--batch-size 100`):
    ```
    python launch_benchmark.py  \
         --model-name mobilenet_v1 \
         --precision int8 \
         --mode inference \
         --framework tensorflow \
         --accuracy-only \
         --batch-size 100  \
         --socket-id 0 \
         --in-graph /home/<user>/mobilenetv1_int8_pretrained_model.pb  \
         --docker-image intelaipg/intel-optimized-tensorflow:1.14 \
         --data-location /home/<user>/imagenet_validation_dataset \
         -- input_height=224 input_width=224 \
         input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
    ```

    Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
    to get additional debug output or change the default output location.

4. The log file is saved to the `models/benchmarks/common/tensorflow/logs` directory,
    or the directory specified by the `--output-dir` arg. Below are examples of
    what the tail of your log file should look like for the different configs.

    Example log tail when benchmarking for throughput:
    ```
    OMP: Info #250: KMP_AFFINITY: pid 682 tid 885 thread 55 bound to OS proc set 83
    OMP: Info #250: KMP_AFFINITY: pid 682 tid 886 thread 56 bound to OS proc set 0
    OMP: Info #250: KMP_AFFINITY: pid 682 tid 884 thread 54 bound to OS proc set 82
    [Running warmup steps...]
    steps = 10, 1830.24507317 images/sec
    [Running benchmark steps...]
    steps = 10, 1841.47811007 images/sec
    steps = 20, 1848.84108679 images/sec
    steps = 30, 1847.84668478 images/sec
    steps = 40, 1849.15354305 images/sec
    steps = 50, 1840.95611001 images/sec
    Ran inference with batch size 240
    Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190409_222536.log
    ```

    Example log tail when benchmarking for latency:
    ```
    OMP: Info #250: KMP_AFFINITY: pid 681 tid 882 thread 53 bound to OS proc set 81
    OMP: Info #250: KMP_AFFINITY: pid 681 tid 884 thread 55 bound to OS proc set 83
    OMP: Info #250: KMP_AFFINITY: pid 681 tid 885 thread 56 bound to OS proc set 0
    [Running warmup steps...]
    steps = 10, 139.81945463 images/sec
    [Running benchmark steps...]
    steps = 10, 140.212074614 images/sec
    steps = 20, 135.230332731 images/sec
    steps = 30, 133.508530685 images/sec
    steps = 40, 135.724816361 images/sec
    steps = 50, 132.714339957 images/sec
    Ran inference with batch size 1
    Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190409_223122.log
    ```

    Example log tail when running for accuracy:
    ```
    Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7009, 0.8933)
    Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7011, 0.8933)
    Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7013, 0.8933)
    Ran inference with batch size 100
    Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190409_223621.log
    ```

## FP32 Inference Instructions

1. The ImageNet dataset is required for testing accuracy and can also be
   used when running benchmarking. If no datset is provided when running
   benchmarking, synthetic data will be used.

   Download the ImageNet dataset and convert it to the TF records format
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

   * Run benchmarking for latency (with `--batch-size 1`, `--checkpoint`
     with a path to the checkpoint file directory, and the `--data-location`
     is optional):
     ```
     python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intelaipg/intel-optimized-tensorflow:1.14 \
         --model-source-dir /home/<user>/tensorflow/models  \
         --batch-size 1 \
         --socket-id 0 \
         --data-location /dataset/Imagenet_Validation \
         --checkpoint /home/<user>/mobilenet_v1_fp32_pretrained_model
     ```
    * Run benchmarking for throughput (with `--batch-size 100`,
      `--checkpoint` with a path to the checkpoint file directory, and
      the `--data-location` is optional):
      ```
      python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intelaipg/intel-optimized-tensorflow:1.14 \
         --model-source-dir /home/<user>/tensorflow/models  \
         --batch-size 100 \
         --socket-id 0 \
         --data-location /dataset/Imagenet_Validation \
         --checkpoint /home/<user>/mobilenet_v1_fp32_pretrained_model
      ```
    * Run benchmarking for accuracy (with `--batch-size 100`, `--accuracy-only` and `--in-graph` with a path to the frozen graph .pb file):
      ```
      python launch_benchmark.py \
         --precision fp32 \
         --model-name mobilenet_v1 \
         --mode inference \
         --framework tensorflow \
         --docker-image intelaipg/intel-optimized-tensorflow:1.14 \
         --model-source-dir /home/<user>/tensorflow/models  \
         --batch-size 100 \
         --accuracy-only \
         --data-location /dataset/Imagenet_Validation \
         --in-graph /home/<user>/mobilenet_v1_fp32_pretrained_model/mobilenet_v1_1.0_224_frozen.pb
      ```
      Note that the `--verbose` or `--output-dir` flag can be added to any of the above
      commands to get additional debug output or change the default output location.

5. The log files for each benchmarking run are saved at the value of `--output-dir`.

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
     Ran inference with batch size 1
     Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_fp32_20190104_200218.log
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
     Ran inference with batch size 100
     Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_fp32_20190104_200512.log
     ```
   * Below is a sample lof file snippet when testing accuracy:
     ```
     Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7104, 0.8999)
     Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7103, 0.8999)
     Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7102, 0.8999)
     Ran inference with batch size 100
     Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_fp32_20190110_211648.log
     ```