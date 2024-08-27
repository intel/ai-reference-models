<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# ResNet50 v1.5 BFloat16 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running ResNet50 v1.5 BFloat16
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

```bash
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph that you downloaded>
```

ResNet50 v1.5 BFloat16 inference can be run to test accuracy, batch inference, or online inference.
Use one of the following examples below, depending on your use case.

* For accuracy run the following command that uses the `DATASET_DIR`, a batch
  size of 100, and the `--accuracy-only` flag:

```bash
python launch_benchmark.py \
  --data-location ${DATASET_DIR} \
  --in-graph ${PRETRAINED_MODEL} \
  --model-name resnet50v1_5 \
  --framework tensorflow \
  --precision bfloat16 \
  --mode inference \
  --batch-size=100 \
  --output-dir ${OUTPUT_DIR} \
  --accuracy-only \
  --docker-image intel/intel-optimized-tensorflow:latest
```

* For batch inference, use the command below that uses the `DATASET_DIR`, a batch
  size of 128, and the `--benchmark-only` flag:

```bash
python launch_benchmark.py \
  --in-graph ${PRETRAINED_MODEL} \
  --model-name resnet50v1_5 \
  --framework tensorflow \
  --precision bfloat16 \
  --mode inference \
  --batch-size=128 \
  --output-dir ${OUTPUT_DIR} \
  --data-location ${DATASET_DIR} \
  --benchmark-only \
  --docker-image intel/intel-optimized-tensorflow:latest
```

* For online inference, use the command below that uses the `DATASET_DIR`, a batch
  size of 1, and the `--benchmark-only` flag:

```bash
python launch_benchmark.py \
  --in-graph ${PRETRAINED_MODEL} \
  --model-name resnet50v1_5 \
  --framework tensorflow \
  --precision bfloat16 \
  --mode inference \
  --batch-size=1 \
  --output-dir ${OUTPUT_DIR} \
  --data-location ${DATASET_DIR} \
  --benchmark-only \
  --docker-image intel/intel-optimized-tensorflow:latest
```

Example log file snippet when testing accuracy:
```bash
...
Iteration time: ... ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7672, 0.9314)
Iteration time: ... ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7672, 0.9314)
Ran inference with batch size 100
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_bfloat16_{timestamp}.log
```

Example log file snippet when testing batch inference:
```bash
...
Iteration 48: ... sec
Iteration 49: ... sec
Iteration 50: ... sec
Average time: ... sec
Batch size = 128
Throughput: ... images/sec
Ran inference with batch size 128
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_bfloat16_{timestamp}.log
```

Example log file snippet when testing online inference:
```bash
...
Iteration 48: ... sec
Iteration 49: ... sec
Iteration 50: ... sec
Average time: ... sec
Batch size = 1
Latency: ... ms
Throughput: ... images/sec
Ran inference with batch size 1
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_bfloat16_{timestamp}.log
```

Batch and online inference can also be run with multiple instances using
`numactl`. The following commands have examples how to do multi-instance runs
using the `--numa-cores-per-instance` argument. Note that these examples are
running with real data (specified by `--data-location ${DATASET_DIR}`).
To use synthetic data, you can omit that argument.

* For multi-instance batch inference, the recommended configuration uses all
  the cores on a socket for each instance (this means that if you have 2 sockets,
  you would be running 2 instances - one per socket) and a batch size of 128.
  ```bash
  python launch_benchmark.py \
    --model-name resnet50v1_5 \
    --precision bfloat16 \
    --mode inference \
    --framework tensorflow \
    --in-graph ${PRETRAINED_MODEL} \
    --data-location ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 128 \
    --numa-cores-per-instance socket \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- warmup_steps=50 steps=1500
  ```

* For multi-instance online inference, the recommended configuration is using
  4 cores per instance and a batch size of 1.
  ```bash
  python launch_benchmark.py \
    --model-name resnet50v1_5 \
    --precision bfloat16 \
    --mode inference \
    --framework tensorflow \
    --in-graph ${PRETRAINED_MODEL} \
    --data-location ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 1 \
    --numa-cores-per-instance 4 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- warmup_steps=50 steps=1500
  ```
