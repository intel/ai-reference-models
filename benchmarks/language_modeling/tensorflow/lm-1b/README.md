# LM-1B 

This document has instructions for how to run LM-1B benchmark for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference for
other platforms are coming later.

## FP32 Inference Instructions

1. Clone [mlperf/inference](https://github.com/mlperf/inference.git) and 
checkout `setInter` branch.
```
git clone https://github.com/mlperf/inference.git
cd mlperf
git checkout setInter
```

To prepare the checkpoint and dataset, run:
```
python inference/cloud/language_modeling/benchmark.py 
```

2. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
git clone https://github.com/IntelAI/models.git
```

3. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo (from step 2).
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, and the checkpoint directory.

Substitute the `--model-source-dir` to `<path_to_mlperf>/inference/cloud/language_modeling`.
Before benchmarking, ensure that you have run the script to prepare checkpoint files and the dataset
from Step 1.

LM-1B can run for latency or throughput
benchmarking. Use one of the following examples below, depending on
your use case.

For latency (using `--socket-id 0` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name lm-1b \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --model-source-dir <path_to_mlperf>/inference/cloud/language_modeling

```

For throughput (using `--socket-id 0` and `--batch-size 1024`):

```
python launch_benchmark.py \
    --model-name lm-1b \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1024 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --model-source-dir <path_to_mlperf>/inference/cloud/language_modeling \
    -- steps=4 \
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

4.  By default, the log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. The user can specify a 
different directory using `--output-dir`.

Example log tail when benchmarking for latency or throughput:
```
Running warmup...
Running benchmark...
Number samples: 4234
Longest latency was: 2.9153692722320557 seconds. Average latency was:2.891982913017273
Perplexity: 40.110043230980665, target is 40.209 .
Ran inference with batch size 1024
```
