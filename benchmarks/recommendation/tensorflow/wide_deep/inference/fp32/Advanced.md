<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# Wide & Deep FP32 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running Wide & Deep FP32
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
dataset, pretrained model, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model>
export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
```

The following commonds are examples on how Wide & Deep can be run:
* Running the model for online inference:
  ```
  python launch_benchmark.py \
    --framework tensorflow \
    --model-source-dir $TF_MODELS_DIR \
    --precision fp32 \
    --mode inference \
    --model-name wide_deep \
    --batch-size 1 \
    --data-location $DATASET_DIR \
    --checkpoint $PRETRAINED_MODEL \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --output-dir $OUTPUT_DIR \
    --verbose
  ```
  The three locations used (model-source-dir, data-location, checkpoint) here,
  works better with docker if they are located in the local disk. The locations
  should be pointed as absolute path.
* Running the model in batch inference mode:
  ```
  python launch_benchmark.py \
    --framework tensorflow \
    --model-source-dir $TF_MODELS_DIR \
    --precision fp32 \
    --mode inference \
    --model-name wide_deep \
    --batch-size 1024 \
    --data-location $DATASET_DIR \
    --checkpoint $PRETRAINED_MODEL \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --output-dir $OUTPUT_DIR \
    --verbose
  ```
  The three locations used (model-source-dir, data-location, checkpoint) here,
  works better with docker if they are located in the local disk. The locations
  should be pointed as absolute path.

