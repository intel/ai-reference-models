# SSD-MobileNet

This document has instructions for how to run SSD-MobileNet for the
following modes/precisions:
- [SSD-MobileNet](#ssd-mobilenet)
  - [Int8 Inference Instructions](#int8-inference-instructions)
  - [FP32 Inference Instructions](#fp32-inference-instructions)

Instructions and scripts for model training and inference
for other precisions are coming later.

## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Clone the [tensorflow/models](https://github.com/tensorflow/models)
repository as `tensorflow-models` at the specified SHA and clone the
[cocoapi repo](git clone https://github.com/cocodataset/cocoapi.git) in
the models directory:
```
$ git clone https://github.com/tensorflow/models.git tensorflow-models
$ cd tensorflow-models
$ git checkout 20da786b078c85af57a4c88904f7889139739ab0
$ git clone https://github.com/cocodataset/cocoapi.git
```
The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

2. Download and preprocess the COCO validation images using the [instructions here](datasets/coco/README.md).
   Be sure to export the $OUTPUT_DIR environment variable.
  
3. Download the pretrained model:

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb
```

4. Clone the [intelai/models](https://github.com/intelai/models) repo
and then run the scripts for online inference performance
or accuracy.
```
$ git clone https://github.com/IntelAI/models.git
$ cd benchmarks
```

Run for online inference where the `--data-location`
is the path to the tf record file that you generated in step 2:
```
python launch_benchmark.py \
    --model-name ssd-mobilenet \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --data-location ${OUTPUT_DIR}/coco_val.record \
    --in-graph /home/<user>/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb \
    --benchmark-only \
    --batch-size 1
```

Or for accuracy where the `--data-location` is the path to
the tf record file that you generated in step 2:
```
python launch_benchmark.py \
    --model-name ssd-mobilenet \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --data-location ${OUTPUT_DIR}/coco_val.record \
    --in-graph /home/<user>/ssdmobilenet_int8_pretrained_model_combinedNMS_s8.pb \
    --accuracy-only \
    --batch-size 1
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

5. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running for online inference:

```
Step 4970: 0.0305020809174 seconds
Step 4980: 0.0294089317322 seconds
Step 4990: 0.0301029682159 seconds
Avg. Duration per Step:0.0300041775227
Avg. Duration per Step:0.0301246762276
Ran inference with batch size 1
Log location outside container: <output directory>/benchmark_ssd-mobilenet_inference_int8_20190417_175418.log
```

And here is a sample log file tail when running for accuracy:

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.53s).
Accumulating evaluation results...
DONE (t=1.10s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.172
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.271
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.183
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.172
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.171
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Ran inference with batch size 1
Log location outside container: <output directory>/benchmark_ssd-mobilenet_inference_int8_20181204_185432.log
```

## FP32 Inference Instructions

1. Clone the `tensorflow/models` repository as `tensorflow-models` with the specified SHA,
since we are using an older version of the models repo for
SSD-MobileNet.

```
$ git clone https://github.com/tensorflow/models.git tensorflow-models
$ cd tensorflow-models
$ git checkout 20da786b078c85af57a4c88904f7889139739ab0
$ git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

2. Follow the TensorFlow models object detection
[installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#installation)
to get your environment setup with the required dependencies.

3. Download and preprocess the COCO validation images using the [instructions here](datasets/coco/README.md).
   Be sure to export the $OUTPUT_DIR environment variable.

4. Download the pretrained model:

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb
```

5. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

```
$ git clone https://github.com/IntelAI/models.git
Cloning into 'models'...
remote: Enumerating objects: 11, done.
remote: Counting objects: 100% (11/11), done.
remote: Compressing objects: 100% (7/7), done.
remote: Total 11 (delta 3), reused 4 (delta 0), pack-reused 0
Receiving objects: 100% (11/11), done.
Resolving deltas: 100% (3/3), done.
```

6. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was just
cloned in the previous step. SSD-MobileNet can be run for testing
online inference or testing accuracy.

To run for online inference, use the following command,
but replace in your path to the processed coco dataset images from step 3
for the `--dataset-location`, the path to the frozen graph that you
downloaded in step 4 as the `--in-graph`, and use the `--benchmark-only`
flag:

```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --data-location ${OUTPUT_DIR}/coco_val.record \
    --in-graph /home/<user>/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --benchmark-only
```

To test accuracy, use the following command but replace in your path to
the tf record file that you generated for the `--data-location`,
the path to the frozen graph that you downloaded in step 4 as the
`--in-graph`, and use the `--accuracy-only` flag:

```
$ python launch_benchmark.py \
    --data-location ${OUTPUT_DIR}/coco_val.record \
    --in-graph /home/<user>/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --accuracy-only
```

8. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running for performance:

```
INFO:tensorflow:Processed 5001 images... moving average latency 37 ms
INFO:tensorflow:Finished processing records
Latency: min = 33.8, max = 6635.9, mean= 38.4, median = 37.2
Ran inference with batch size -1
Log location outside container: {--output-dir value}/benchmark_ssd-mobilenet_inference_fp32_20190130_225108.log
```

Below is a sample log file tail when testing accuracy:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Ran inference with batch size -1
Log location outside container: {--output-dir value}/benchmark_ssd-mobilenet_inference_fp32_20190123_225145.log
```
