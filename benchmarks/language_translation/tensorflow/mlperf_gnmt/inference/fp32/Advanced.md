<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# MLPerf GNMT FP32 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running MLPerf GNMT FP32
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
If you are going to run using docker, copy the `tensorflow-addons` wheel that you built
during the [model setup](README.md#run-the-model) to the model zoo's `mlperf_gnmt` directory:
```
cp <tensorflow-addons repo>/artifacts/tensorflow_addons-*.whl <model zoo directory>/models/language_translation/tensorflow/mlperf_gnmt
```

Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables for the dataset, checkpoint
directory, and an output directory where log files will be written.
```
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model frozen graph .pb file>
```

MLPerf GNMT inference can be run in three different modes:

* For online inference, use the following command (with `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):
  ```
  python launch_benchmark.py \
    --model-name mlperf_gnmt \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 1 \
    --socket-id 0 \
    --data-location $DATASET_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --in-graph $PRETRAINED_MODEL \
    --output-dir $OUTPUT_DIR \
    --benchmark-only
  ```
* For batch inference, use the following command (with `--benchmark-only`, `--socket-id 0` and `--batch-size 32`):
  ```
  python launch_benchmark.py \
    --model-name mlperf_gnmt \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 32 \
    --socket-id 0 \
    --data-location $DATASET_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --in-graph $PRETRAINED_MODEL \
    --output-dir $OUTPUT_DIR \
    --benchmark-only
  ```
* For accuracy testing, use the following command (with `--accuracy_only`, `--socket-id 0` and `--batch-size 32`):
  ```
  python launch_benchmark.py \
    --model-name mlperf_gnmt \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 32 \
    --socket-id 0 \
    --data-location $DATASET_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --in-graph $PRETRAINED_MODEL \
    --output-dir $OUTPUT_DIR \
    --accuracy-only
  ```

