<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# SSD-ResNet34 Int8 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running SSD-ResNet34 Int8
inference, which provides more control over the individual parameters that
are used. For more information on using [`/benchmarks/launch_benchmark.py`](/benchmarks/launch_benchmark.py),
see the [launch benchmark documentation](/docs/general/tensorflow/LaunchBenchmark.md).

Prior to using these instructions, please follow the setup instructions from
the model's [README](README.md) and/or the
[AI Kit documentation](/docs/general/tensorflow/AIKit.md) to get your environment
setup (if running on bare metal) and download the dataset, pretrained model, etc.
If you are using AI Kit, please exclude the `--docker-image` flag from the
commands below, since you will be running the the TensorFlow conda environment
instead of docker.

<!-- 55. Docker arg -->
Any of the `launch_benchmark.py` commands below can be run on bare metal by
removing the `--docker-image` arg. Ensure that you have all of the
[required prerequisites installed](README.md#run-the-model) in your environment
before running without the docker container.

If you are new to docker and are running into issues with the container,
see [this document](/docs/general/docker.md) for troubleshooting tips.

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

SSD-ResNet34 can be run for testing batch or online inference, or testing accuracy.

To run for batch and online inference, use the command below. If you are
running with docker, you will also need to provide the `ssd-resnet-benchmarks`
path for `volume` flag. To run without docker, omit the `--docker-image` and
`--volume` flags. By default it runs with input size 300x300, you may
add `-- input-size=1200` flag to run benchmark with input size 1200x1200.
Use the 300x300 or 1200x1200 pretrained model, depending on the input size.
Optionally, you can also specify the number of `warmup-steps` and `steps` as
shown in the example below, the default values are `warmup-steps=200` and `steps=800`.
```
# benchmarks with input size 300x300
python launch_benchmark.py \
    --in-graph ${PRETRAINED_MODEL} \
    --model-source-dir ${TF_MODELS_DIR} \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --output-dir ${OUTPUT_DIR} \
    --benchmark-only \
    -- warmup-steps=50 steps=200
```

To run the accuracy test, use the command below. By default it runs with
input size 300x300, you may add `-- input-size=1200` flag to run the test with
input size 1200x1200. Use the 300x300 or 1200x1200 pretrained model,
depending on the input size. To run without docker, omit the `--docker-image` and
`--volume` flags.
```
# accuracy test with input size 1200x1200
python launch_benchmark.py \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --model-source-dir ${TF_MODELS_DIR} \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --output-dir ${OUTPUT_DIR} \
    --accuracy-only \
    -- input-size=1200
```

The log file is saved to the `${OUTPUT_DIR}`.

Below is a sample log file tail when testing performance:
```
Batchsize: 1
Time spent per BATCH:    ... ms
Total samples/sec:    ... samples/s
```

Below is a sample log file tail when testing accuracy:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.204
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.208
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.051
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.210
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.083
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.484
Current AP: 0.20408
```

