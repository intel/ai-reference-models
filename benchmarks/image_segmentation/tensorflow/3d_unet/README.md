# 3D U-Net

This document has instructions for how to run 3D U-Net for the following
modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions

1. Follow the instructions at the [3D U-Net repository](https://github.com/ellisdg/3DUnetCNN)
   for [downloading and preprocessing the BraTS dataset](https://github.com/ellisdg/3DUnetCNN/blob/ff5953b3a407ded73a00647f5c2029e9100e23b1/README.md#tutorial-using-brats-data-and-python-3).
   The directory that contains the preprocessed dataset files will be
   passed to the launch script when running the benchmarking script.

2. Clone this [intelai/models](https://github.com/IntelAI/models)
   repository:
   ```
   $ git clone https://github.com/IntelAI/models.git
   ```

   This repository contains the scripts that we will use for running
   benchmarks as well as the Intel-Optimized 3D U-Net model code.

3. Download the pre-trained model from the
   [3DUnetCNN](https://github.com/ellisdg/3DUnetCNN/blob/master/README.md#pre-trained-models)
   repository. The one we are using in this example is the one trained
   using the one called the "Original U-Net" model, trained using the
   BraTS 2017 data.

4. Navigate to the `benchmarks` directory in your clone of the
   [intelai/models](https://github.com/IntelAI/models) repo from step 2.
   This directory contains the `launch_benchmarks.py` script, which we
   will use to run benchmarking. The script takes parameters that
   specify which model to run as well a path to the preprocessed BraTS
   data from step 1 as the `--data-location` and the path to the
   pretrained 3D U-Net model that was downloaded in step 3 as the
   `--in-graph`.

   To run benchmarking for throughput and latency, use the following
   command (replace in your `--data-location` and `--in-graph`):

   ```
   $ cd /home/myuser/intelai/models/benchmarks

   $ python launch_benchmark.py \
        --precision fp32 \
        --model-name 3d_unet \
        --mode inference \
        --framework tensorflow \
        --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl-py3 \
        --in-graph /home/$USER/tumor_segmentation_model.h5 \
        --data-location /home/$USER/3dunet_data/BraTS \
        --batch-size 1 \
        --socket-id 0
   ```

   Note that the `--verbose` or `--output-dir` flag can be added to the above command
   to get additional debug output or change the default output location.

5. Log files are located in the `--output-dir` directory.

   Below is an example tail of log file when benchmarking throughput and
   latency:

   ```
   Loading pre-trained model
   Time spent per BATCH: 12922.1132 ms
   Total samples/sec: 0.0774 samples/s
   lscpu_path_cmd = command -v lscpu
   lscpu located here: b'/usr/bin/lscpu'
   Ran inference with batch size 1
   Log location outside container: <output directory>/benchmark_3d_unet_inference_fp32_20190116_234659.log
   ```