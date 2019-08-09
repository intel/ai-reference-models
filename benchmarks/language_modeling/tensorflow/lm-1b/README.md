# LM-1B 

This document has instructions for how to run LM-1B for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model training and inference for
other platforms are coming later.

## FP32 Inference Instructions

1. Store the path to the current directory and clone [mlperf/inference](https://github.com/mlperf/inference.git)
with the current SHA from master of the repo on 6/26/2019:
```
$ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
$ pushd $MODEL_WORK_DIR

$ git clone https://github.com/mlperf/inference.git
$ cd inference
$ git checkout 41eb3e489233e83e544cd25148aca177b95d7bea
```

To prepare the checkpoint and dataset, run the `benchmark.py` script
from the mlperf inference repo. Since this requires python3 and
TensorFlow to be installed, the following instructions show how to run
a docker container with your cloned mlperf inference repo mounted as a
volume:
```
$ docker run --volume $MODEL_WORK_DIR/inference:/inference -it gcr.io/deeplearning-platform-release/tf-cpu.1-14 /bin/bash
```
In the docker container, run:
```
$ cd /inference/others/cloud/language_modeling/
$ python3 benchmark.py
```

2. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

3. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo (from step 2).
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, and the checkpoint directory.

Substitute the `--model-source-dir` to `<path_to_mlperf>/inference/cloud/language_modeling`.
Before running, ensure that you have run the script to prepare checkpoint files and the dataset
from Step 1.

LM-1B can run for online or batch inference. Use one of the following examples below, depending on
your use case.

For online inference (using `--socket-id 0` and `--batch-size 1`):

```
$ python launch_benchmark.py \
    --model-name lm-1b \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --model-source-dir $MODEL_WORK_DIR/inference/others/cloud/language_modeling

```

For batch inference (using `--socket-id 0` and `--batch-size 1024`):

```
$ python launch_benchmark.py \
    --model-name lm-1b \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1024 \
    --socket-id 0 \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --model-source-dir $MODEL_WORK_DIR/inference/others/cloud/language_modeling \
    -- steps=4 \
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

4.  By default, the log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. The user can specify a 
different directory using `--output-dir`.

Example log tail when running for online or batch inference:
```
Running warmup...
Running benchmark...
Number samples: 4234
Longest latency was: 2.9153692722320557 seconds. Average latency was:2.891982913017273
Perplexity: 40.110043230980665, target is 40.209 .
Ran inference with batch size 1024
```

5. To return to where you started from:
```
$ popd
```
