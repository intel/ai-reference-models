<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# Faster R-CNN Int8 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running Faster R-CNN Int8
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
dataset, pretrained model frozen graph, the TensorFlow models repo, and an output
directory where log files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export PRETRAINED_MODEL=<path to the pretrained model frozen graph file>
export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
export OUTPUT_DIR=<directory where log files will be saved>
```
Run batch and online inference using the following command.
```
python launch_benchmark.py \
 --data-location ${DATASET_DIR} \
 --model-source-dir ${TF_MODELS_DIR} \
 --model-name faster_rcnn \
 --framework tensorflow \
 --precision int8 \
 --mode inference \
 --socket-id 0 \
 --in-graph ${PRETRAINED_MODEL} \
 --docker-image intel/intel-optimized-tensorflow:1.15.2 \
 --output-dir ${OUTPUT_DIR} \
 --benchmark-only \
 -- number_of_steps=5000
```

Test accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located and the `--in-graph` is
the pre-trained graph model:
```
python launch_benchmark.py \
 --model-name faster_rcnn \
 --mode inference \
 --precision int8 \
 --framework tensorflow \
 --socket-id 0 \
 --docker-image intel/intel-optimized-tensorflow:1.15.2 \
 --model-source-dir ${TF_MODELS_DIR} \
 --data-location ${DATASET_DIR}/coco_val.record \
 --in-graph ${PRETRAINED_MODEL} \
 --output-dir ${OUTPUT_DIR} \
 --accuracy-only
```

Output files and logs are saved to the `${OUTPUT_DIR}` directory.

Below is a sample log file tail when running for batch and online inference:
```
Step 4970: ... seconds
Step 4980: ... seconds
Step 4990: ... seconds
Avg. Duration per Step: ...
Log location outside container: ${OUTPUT_DIR}/benchmark_faster_rcnn_inference_int8_20190117_232539.log
```

And here is a sample log file tail when running for accuracy:
```
Accumulating evaluation results...
DONE (t=1.34s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.310
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.479
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.351
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.310
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.267
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.372
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Ran inference with batch size -1
Log location outside container: ${OUTPUT_DIR}/benchmark_faster_rcnn_inference_int8_20190117_231937.log
```
