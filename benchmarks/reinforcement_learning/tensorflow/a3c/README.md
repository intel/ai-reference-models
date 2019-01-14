## Asynchronous Advantage Actor-Critic (A3C) ##

This document has instructions for how to run A3C for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference.

## FP32 Inference Instructions

1. Clone and install the [Arcade-Learning-Environment](https://github.com/miyosuda/Arcade-Learning-Environment) repository.
`Arcade-Learning-Environment` is used as an external model repository for dependencies.
```
$ git clone https://github.com/miyosuda/Arcade-Learning-Environment.git
$ cd Arcade-Learning-Environment
$ cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF .
$ make -j 8
$ pip install .

```

2. A link to download the pre-trained model is coming soon.

3. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone git@github.com:IntelAI/models.git
```

This repository includes launch scripts for running benchmarks and the
an optimized version of the A3C model code.

4. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 3.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the external model directory
for `--model-source-dir` (from step 1) and `--checkpoint` (from step 2).


Run benchmarking for throughput and latency with `--batch-size=1` :
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --model-source-dir /home/myuser/Arcade-Learning-Environment \
    --model-name a3c \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 1 \
    --socket-id 0 \
    --checkpoint /home/myuser/a3c_fp32_checkpoint \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```

5. Log files are located at:
`intelai/models/benchmarks/common/tensorflow/logs`.

Below is a sample log file tail when running benchmarking for throughput
and latency:
```
Running ROM file...
Random seed is 0
Batch size: 1
('Batches number:', 2000)
Time spent per BATCH: 2.2826 ms
Total samples/sec: 438.0949 samples/s
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: 
Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_a3c_inference_fp32_20190108_192920.log
```
