<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# Wide and Deep using a large dataset FP32 training - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running Wide and Deep using a large dataset FP32
training, which provides more control over the individual parameters that
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
dataset (where train.csv and eval.csv are located), an output directory where log
files will be written, and optionally a directory where checkpoint files can
be read and written from.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset directory>
export OUTPUT_DIR=<directory where the logs and the saved model will be written>
export CHECKPOINT_DIR=<Optional directory where checkpoint files will be read and written>
```

Train the model (The model will be trained for 10 epochs if -- steps is not specified)
```
python launch_benchmark.py \
  --model-name wide_deep_large_ds \
  --precision fp32 \
  --mode training  \
  --framework tensorflow \
  --batch-size 512 \
  --data-location ${DATASET_DIR} \
  --checkpoint ${CHECKPOINT_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --docker-image intel/intel-optimized-tensorflow:latest
```
Once the training completes successfully the path of checkpoint files and saved_model.pb will be printed as shown below
```
INFO:tensorflow:SavedModel written to: $OUTPUT_DIR/temp-1602670603/saved_model.pb
Using TensorFlow version 2.4.0
Begin training and evaluation
Saving model checkpoints to $CHECKPOINT_DIR
****Computing statistics of train dataset*****
estimator built
fit done
evaluate done
Model exported to $OUTPUT_DIR
```

