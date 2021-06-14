<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# MobileNet V1 Int8 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running MobileNet V1 Int8
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

MobileNet V1 Int8 inference can be run to test accuracy, batch inference, or online inference.
Use one of the following examples below, depending on your use case.

* For accuracy run the following command that uses the `DATASET_DIR`, a batch
  size of 100, and the `--accuracy-only` flag:

```
python launch_benchmark.py \
  --model-name mobilenet_v1 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --accuracy-only \
  --batch-size 100  \
  --socket-id 0 \
  --output-dir ${OUTPUT_DIR} \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -- input_height=224 input_width=224 \
  input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
```

* For batch inference, use the command below that uses the `DATASET_DIR`, a batch 
  size of 240, and the `--benchmark-only` flag.

```
python launch_benchmark.py \
  --model-name mobilenet_v1 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --benchmark-only \
  --batch-size 240  \
  --socket-id 0 \
  --output-dir ${OUTPUT_DIR} \
  --in-graph ${PRETRAINED_MODEL} \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -- input_height=224 input_width=224 warmup_steps=10 steps=50 \
  input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
```

* For online inference, use the command below that uses the `DATASET_DIR`, a batch 
  size of 1, and the `--benchmark-only` flag.
  
```
python launch_benchmark.py \
  --model-name mobilenet_v1 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --benchmark-only \
  --batch-size 1  \
  --socket-id 0 \
  --output-dir ${OUTPUT_DIR} \
  --in-graph ${PRETRAINED_MODEL} \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -- input_height=224 input_width=224 warmup_steps=10 steps=50 \
  input_layer="input" output_layer="MobilenetV1/Predictions/Reshape_1"
```

Example log tail when running for accuracy:
```
Iteration time: ... ms
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7014, 0.8935)
Iteration time: ... ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7014, 0.8934)
Iteration time: ... ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7014, 0.8934)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190523_164955.log
```

Example log tail when running for batch inference:
```
[Running warmup steps...]
steps = 10, ... images/sec
[Running benchmark steps...]
steps = 10, ... images/sec
steps = 20, ... images/sec
steps = 30, ... images/sec
steps = 40, ... images/sec
steps = 50, ... images/sec
Ran inference with batch size 240
Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190523_164626.log
```

Example log tail when running for online inference:
```
[Running warmup steps...]
steps = 10, ... images/sec
[Running benchmark steps...]
steps = 10, ... images/sec
steps = 20, ... images/sec
steps = 30, ... images/sec
steps = 40, ... images/sec
steps = 50, ... images/sec
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_mobilenet_v1_inference_int8_20190523_164348.log
```

Batch and online inference can also be run with multiple instances using
`numactl`. The following commands have examples how to do multi-instance runs
using the `--numa-cores-per-instance` argument. Note that these examples are
running with synthetic data (to use real data, you can add `--data-location ${DATASET_DIR}`).
Your output may vary from what's seen below, depending on the number of
cores on your system.

* For multi-instance batch inference, the recommended configuration uses all
  the cores on a socket for each instance (this means that if you have 2 sockets,
  you would be running 2 instances - one per socket) and a batch size of 56.
  ```  
  python launch_benchmark.py \
    --model-name mobilenet_v1 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 56 \
    --numa-cores-per-instance socket \
    --output-dir ${OUTPUT_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- input_height=224 input_width=224 warmup_steps=500 steps=1000 \
    input_layer='input' output_layer='MobilenetV1/Predictions/Reshape_1'
  ```
  
* For multi-instance online inference, the recommended configuration is using
  4 cores per instance and a batch size of 1.
  ```
  python launch_benchmark.py \
    --model-name mobilenet_v1 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --numa-cores-per-instance 4 \
    --output-dir ${OUTPUT_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- input_height=224 input_width=224 warmup_steps=500 steps=1000 \
    input_layer='input' output_layer='MobilenetV1/Predictions/Reshape_1'
  ```

