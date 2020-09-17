## Mask R-CNN ##

This document has instructions for how to run Mask R-CNN for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference.

## FP32 Inference Instructions

1. Download the [MS COCO 2014 dataset](http://cocodataset.org/#download).

2. Clone the [Mask R-CNN model repository](https://github.com/matterport/Mask_RCNN).
It is used as external model directory for dependencies.
```
$ https://github.com/matterport/Mask_RCNN.git
$ cd Mask_RCNN
```

3. Download pre-trained COCO weights `mask_rcnn_coco.h5)` from the
[Mask R-CNN repository release page](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5),
and place it in the `MaskRCNN` directory (from step 2) .
```
$ wget -q https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 
```

4. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ https://github.com/IntelAI/models.git
```

This repository includes launch scripts for running benchmarks and the
an optimized version of the Mask R-CNN model code.

5. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 4.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the external model directory
for `--model-source-dir` (from step 2) and `--data-location` (from step 1).


Run benchmarking for throughput and latency with `--batch-size=1` :
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --model-source-dir /home/<user>/Mask_RCNN \
    --model-name maskrcnn \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 1 \
    --socket-id 0 \
    --data-location /home/<user>/COCO2014 \
    --docker-image intelaipg/intel-optimized-tensorflow:1.15.2 
```

5. Log files are located at:
`intelai/models/benchmarks/common/tensorflow/logs`, or you can provide specific location
using `--output-dir` 

Below is a sample log file tail when running benchmarking for throughput
and latency:
```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.19s).
Accumulating evaluation results...
DONE (t=0.16s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.612
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.373
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
Batch size: 1
Time spent per BATCH: 734.5615 ms
Total samples/sec: 1.3614 samples/s
Total time:  43.79594111442566
Ran inference with batch size 1
Log file location: {--output-dir value}/benchmark_maskrcnn_inference_fp32_20200917_164707.log
```
