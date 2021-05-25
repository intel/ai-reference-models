<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export MODEL_SRC_DIR=<path to the Mask RCNN models repo>
```

<model name> <precision> <mode> can be run for throughput and latency with `--batch-size=1`:

```
python launch_benchmark.py \
  --model-source-dir ${MODEL_SRC_DIR} \
  --model-name maskrcnn \
  --framework tensorflow \
  --precision fp32 \
  --mode inference \
  --batch-size 1 \
  --socket-id 0 \
  --data-location ${DATASET_DIR} \
  --docker-image intel/intel-optimized-tensorflow:1.15.2 \
  --output-dir ${OUTPUT_DIR}
```

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
Time spent per BATCH: ... ms
Total samples/sec: ... samples/s
Total time:  ...
Ran inference with batch size 1
Log file location: {--output-dir value}/benchmark_maskrcnn_inference_fp32_20200917_164707.log
```
