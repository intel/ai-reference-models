<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
coco validation dataset, TensorFlow models repo, pretrained model frozen graph,
and an output directory where log files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<directory with the validation-*-of-* files (for accuracy testing only)>
export TF_MODELS_DIR=<path to the TensorFlow Models repo>
export PRETRAINED_MODEL=<path to the 300x300 or 1200x1200 pretrained model pb file>
export OUTPUT_DIR=<directory where log files will be written>
```

SSD-ResNet34 can be run for batch and online inference, or accuracy.

To run for batch and online inference, use the command below.
If you are running using docker, you will need to provide the `ssd-resnet-benchmarks`
path for `volume` flag. To run without docker, omit the `--docker-image` and
`--volume` flags. By default it runs with input size 300x300, you
may add `-- input-size=1200` flag to run benchmark with input size 1200x1200.
Use the 300x300 or 1200x1200 pretrained model, depending on the input size.
Optionally, you can also specify the number of `warmup-steps` and `steps` as
shown in the example below, the default values are `warmup-steps=200` and `steps=800`.
```
# benchmarks with input size 1200x1200
python launch_benchmark.py \
    --in-graph ${PRETRAINED_MODEL} \
    --model-source-dir ${TF_MODELS_DIR} \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image <docker image> \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --output-dir ${OUTPUT_DIR} \
    --benchmark-only \
    -- input-size=1200 warmup-steps=50 steps=200
```

To run the accuracy test, use the command below.
By default it runs with  input size 300x300, you may add `-- input-size=1200`
flag last to run the test with input size 1200x1200. Use the 300x300 or
1200x1200 pretrained model, depending on the input size. To run without docker,
omit the `--docker-image` and `--volume` flags.
```
# accuracy test with input size 300x300
python launch_benchmark.py \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --model-source-dir ${TF_MODELS_DIR} \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image <docker image> \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --output-dir ${OUTPUT_DIR} \
    --accuracy-only
```

The log file is saved to the value of `${OUTPUT_DIR}`.

Below is a sample log file tail when running for performance:
```
Batchsize: 1
Time spent per BATCH:    ... ms
Total samples/sec:    ... samples/s
```

Below is a sample log file tail when testing accuracy:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.216
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.494
Current AP: 0.21082
```
