<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# ResNet101 Int8 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running ResNet101 Int8
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
dataset, pretrained model frozen graph, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph that you downloaded>
```

ResNet101 Int8 inference can be run to test accuracy, batch inference, or online inference.
Use one of the following examples below, depending on your use case.

* For accuracy run the following command that uses the `DATASET_DIR`, a batch
  size of 100, and the `--accuracy-only` flag:

```
python launch_benchmark.py \
  --model-name resnet101 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --accuracy-only \
  --batch-size 100 \
  --socket-id 0 \
  --docker-image intel/intel-optimized-tensorflow:latest
```

* When running for performance, it is optional to specify the
  number of `warmup_steps` and `steps` as extra args, as shown in the
  commands below. If these values are not specified, the script will
  default to use `warmup_steps=40` and `steps=100`. For batch inference, 
  use the command below that uses the `DATASET_DIR`, a batch size of 128,
  `warmup_steps=50`, and `steps=500`.

```
python launch_benchmark.py \
  --model-name resnet101 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size 128 \
  --socket-id 0 \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -- warmup_steps=50 steps=500
```

* For online inference, use the command below that uses the `DATASET_DIR`, a batch 
  size of 1, `warmup_steps=50`, and `steps=500`.
  
```
python launch_benchmark.py \
  --model-name resnet101 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size 1 \
  --docker-image intel/intel-optimized-tensorflow:latest \
  --socket-id 0 \
  -- warmup_steps=50 steps=500
```

Example log file snippet when testing accuracy:
```
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7690, 0.9304)
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7691, 0.9305)
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7691, 0.9305)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190104_205838.log
```

Example log file snippet when testing batch inference:
```
...
steps = 470, ... images/sec
steps = 480, ... images/sec
steps = 490, ... images/sec
steps = 500, ... images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190223_192438.log
```

Example log file snippet when testing online inference:
```
...
steps = 470, ... images/sec
steps = 480, ... images/sec
steps = 490, ... images/sec
steps = 500, ... images/sec
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190223_191406.log
```

