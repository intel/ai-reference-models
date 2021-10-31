<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# ResNet50 v1.5 FP32 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running ResNet50 v1.5 FP32
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

ResNet50 v1.5 FP32 inference can be run to test accuracy, batch inference, or online inference.
Use one of the following examples below, depending on your use case.

* For accuracy run the following command that uses the `DATASET_DIR`, a batch
  size of 100, and the `--accuracy-only` flag:

```
python launch_benchmark.py \
  --model-name resnet50v1_5 \
  --precision fp32 \
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

* For batch inference, use the command below that uses the `DATASET_DIR` and a batch 
  size of 128.

```
python launch_benchmark.py \
  --model-name resnet50v1_5 \
  --precision fp32 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size=128 \
  --socket-id 0 \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -- warmup_steps=50 steps=1500
```

* For online inference, use the command below that uses the `DATASET_DIR` and a batch 
  size of 1.
  
```
python launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision=fp32 \
  --mode=inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size=1 \
  --socket-id 0 \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -- warmup_steps=50 steps=1500
```

Example log file snippet when testing accuracy:
```
...
Iteration time: ... ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7648, 0.9308)
Ran inference with batch size 100
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp32_{timestamp}.log
```

Example log file snippet when testing batch inference:
```
Iteration 1: ... sec
Iteration 2: ... sec
Iteration 3: ... sec
...
Iteration 388: ... sec
Iteration 389: ... sec
Iteration 390: ... sec
Average time: ... sec
Batch size = 128
Throughput: ... images/sec
Ran inference with batch size 128
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp32_{timestamp}.log
```

Example log file snippet when testing online inference:
```
Iteration 1: ... sec
Iteration 2: ... sec
Iteration 3: ... sec
...
Iteration 1498: ... sec
Iteration 1499: ... sec
Iteration 1500: ... sec
Average time: ... sec
Batch size = 1
Latency: ... ms
Throughput: ... images/sec
Ran inference with batch size 1
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp32_{timestamp}.log
```

Batch and online inference can also be run with multiple instances using
`numactl`. The following commands have examples how to do multi-instance runs
using the `--numa-cores-per-instance` argument. Note that these examples are
running with real data (specified by `--data-location ${DATASET_DIR}`).
To use synthetic data, you can omit that argument.

* For multi-instance batch inference, the recommended configuration uses all
  the cores on a socket for each instance (this means that if you have 2 sockets,
  you would be running 2 instances - one per socket) and a batch size of 128.
  ```  
  python launch_benchmark.py \
    --model-name resnet50v1_5 \
    --precision fp32 \
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
  ```
  python launch_benchmark.py \
    --model-name resnet50v1_5 \
    --precision fp32 \
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

