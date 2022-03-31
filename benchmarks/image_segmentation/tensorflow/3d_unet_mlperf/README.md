# 3D-Unet on BraTS 2019 dataset (fold 1)

This document has instructions for how to run MLPerf 3D U-Net model for the following
modes/precisions:
* [FP32 inference](#fp32-inference-instructions)
* [BFloat16 inference](#bfloat16-inference-instructions)
* [Int8 inference](#int8-inference-instructions)

## FP32 Inference Instructions

1. Please download [Brats 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) 
   separately and unzip the dataset. The directory that contains the dataset files will be
   passed to the launch script when running the benchmarking script.

2. Download the pre-trained model from the
   [3DUnetCNN](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3dunet_dynamic_ndhwc.pb).
   In this example, we are using the model,
   trained using the fold 1 BRATS 2019 data.
   The validation files have been copied from [here](https://github.com/mlcommons/inference/tree/r0.7/vision/medical_imaging/3d-unet/folds)

3. Clone this [intelai/models](https://github.com/IntelAI/models)
   repository:
   ```
   git clone https://github.com/IntelAI/models.git
   ```
   This repository contains the scripts that we will use for running
   benchmarks as well as the Intel-Optimized 3D-Unet model code.

4. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance and/or calculate the accuracy. 

   * Evaluate the model accuracy: For evaluating accuracy the required parameters include: the `Brats 2019` dataset location (from step 1), the pre-trained `3dunet_dynamic_ndhwc.pb` input graph file (from step 2), and the `--accuracy-only` flag.

   ```
   cd /home/<user>/models/benchmarks

   python launch_benchmark.py \
      --data-location /home/<user>/MICCAI_BraTS_2019_Data_Training \
      --in-graph /home/<user>/3dunet_dynamic_ndhwc.pb \
      --model-name 3d_unet_mlperf \
      --framework tensorflow \
      --precision fp32 \
      --mode inference \
      --batch-size 1 \
      --accuracy-only \
      --docker-image intel/intel-optimized-tensorflow:latest
   ```

   The tail of the log output when the accuracy run completes should look
   something like this:
   ```
   Running evaluation...
   WARNING! Some files in folder_gt were not predicted (not present in folder_predicted)!
   Loading evaluation summary...
   Accuracy: mean = 0.85300, whole tumor = 0.9141, tumor core = 0.8679, enhancing tumor = 0.7770
   Done!
   ```
   * Evaluate the model performance: Currently, for performance evaluation dummy data is used. The required parameters include: the pre-trained `3dunet_dynamic_ndhwc.pb` input graph file (from step 2) and the `--benchmark-only` flag. Optionally, you can also specify the number of `warmup_steps` and `steps` as shown in the example below. If these optional arguments are not specified, the script will use the default values `warmup_steps=10` and `steps=50`.
   ```
   cd /home/<user>/models/benchmarks

   python launch_benchmark.py \
      --in-graph /home/<user>/3dunet_dynamic_ndhwc.pb \
      --model-name 3d_unet_mlperf \
      --framework tensorflow \
      --precision fp32 \
      --mode inference \
      --batch-size 1 \
      --benchmark-only \
      --docker-image intel/intel-optimized-tensorflow:latest \
      -- warmup_steps=20 steps=100
   ```  
   The tail of the log output when the performance run completes should look
   something like this:
   ```
   Inference with dummy data
   Iteration 0
   Iteration 10
   Average time for step: ... sec
   Throughput: ... samples/sec
   Latency: ... ms
   Ran inference with batch size 1
   Log file location: {--output-dir value}/benchmark_3d_unet_mlperf_inference_fp32_{timestamp}.log
   ```

   Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
   to get additional debug output or change the default output location.

#### Run FP32 Inference on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/tensorflow/Windows.md).
Then, install the model dependencies in `models/benchmarks/image_segmentation/tensorflow/3d_unet_mlperf/requirements.txt` and `matplotlib`.

Evaluate the model performance:
Currently, for performance evaluation dummy data is used. The required parameters include: the pre-trained `3dunet_dynamic_ndhwc.pb` input graph file (from step 2) and the `--benchmark-only` flag.
Optionally, you can also specify the number of `warmup_steps` and `steps` as shown in the example below.
If these optional arguments are not specified, the script will use the default values `warmup_steps=10` and `steps=50`.

Using `cmd.exe`, run:
```
python launch_benchmark.py ^
--in-graph <path to pretrained model>\\3dunet_dynamic_ndhwc.pb ^
--model-name 3d_unet_mlperf ^
--framework tensorflow ^
--precision fp32 ^
--mode inference ^
--batch-size 1 ^
--benchmark-only ^
-- warmup_steps=20 steps=100
```

## BFloat16 Inference Instructions
(Experimental)

3D-Unet BFloat16 inference depends on Auto-Mixed_precision to convert graph from FP32 to BFloat16 online.
The instructions are same as FP32 inference instructions, except you need to change the `--precision=fp32` to `--precision=bfloat16` in the above commands.

## Int8 Inference Instructions
Download the pre-trained model from the [3DUnetCNN](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/3dunet_int8_fully_quantized_perchannel.pb).

The rest of 3D-Unet Int8 inference instructions are same as FP32 inference instructions, except you need use `--precision=int8` and  `--in-graph=/home/<user>/3dunet_int8_fully_quantized_perchannel.pb` in the above commands.
