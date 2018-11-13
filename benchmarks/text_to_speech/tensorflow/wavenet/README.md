# WaveNet

This document has instructions for how to run WaveNet for the following
modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference
other platforms are coming later.

## FP32 Inference Instructions

1. Clone the [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
repo and get pull request #352 for the CPU optimizations.  The path to
the cloned repo will be passed as the model source directory when
running the benchmarking script.

```
$ git clone git@github.com:ibab/tensorflow-wavenet.git
Cloning into 'tensorflow-wavenet'...
remote: Enumerating objects: 1, done.
remote: Counting objects: 100% (1/1), done.
remote: Total 918 (delta 0), reused 0 (delta 0), pack-reused 917
Receiving objects: 100% (918/918), 342.32 KiB | 0 bytes/s, done.
Resolving deltas: 100% (528/528), done.

$ cd tensorflow-wavenet/

$ git fetch origin pull/352/head:cpu_optimized
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
remote: Total 4 (delta 3), reused 3 (delta 3), pack-reused 1
Unpacking objects: 100% (4/4), done.
From github.com:ibab/tensorflow-wavenet
 * [new ref]         refs/pull/352/head -> cpu_optimized

$ git checkout cpu_optimized

$ pwd
```

2. Clone this [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking, as well as
checkpoint files for a pre-trained model.  After cloning the repo,
navigate to the benchmarks directory, which is where the launch script
is located.

```
$ git clone git@github.com:IntelAI/models.git

$ cd models/benchmarks
```

3. Start benchmarking by executing the launch script and passing args
specifying that we are running wavenet fp32 inference using TensorFlow,
along with a dockerfile that includes Intel Optimizations for TensorFlow
and the path to the model source dir (from step 1) and the checkpoint
files.  We are also passing a couple of extra model args for wavenet:
the name of the checkpoint to use and the sample number.

```
python launch_benchmark.py \
	--platform fp32 \
	-m wavenet \
	--mode inference \
	--framework tensorflow \
        --single-socket \
        --num-cores 1 \
	--docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
	--model-source-dir /home/myuser/wavenet/tensorflow-wavenet \
	--checkpoint /home/myuser/models/benchmarks/text_to_speech/tensorflow/wavenet/inference/fp32/checkpoints \
    -- checkpoint_name=model.ckpt-99 sample=8510
```

4.  The logs are displayed in the console output as well as saved to a
file at models/benchmarks/common/tensorflow/logs/benchmark_wavenet_inference.log

The tail of the log should look something like this:

```
Time per 500 Samples: 1.470617 sec
Samples / sec: 339.993337
msec / sample: 2.941234
Sample: 7500
Time per 500 Samples: 1.454286 sec
Samples / sec: 343.811360
msec / sample: 2.908572
Sample: 8000
Time per 500 Samples: 1.464048 sec
Samples / sec: 341.518879
msec / sample: 2.928096
Sample: 8500

Average Throughput of whole run: Samples / sec: 304.013303
Average Latency of whole run: msec / sample: 3.289330
Finished generating. The result can be viewed in TensorBoard.
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
current path: /workspace/benchmarks
search path: /workspace/benchmarks/*/tensorflow/wavenet/inference/fp32/model_init.py
Using model init: /workspace/benchmarks/text_to_speech/tensorflow/wavenet/inference/fp32/model_init.py
Received these standard args: Namespace(batch_size=-1, checkpoint='/checkpoints', data_location=None, framework='tensorflow', input_graph=None, mode='inference', model_args=[], model_name='wavenet', model_source_dir='/workspace/models', num_cores=1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, verbose=True)
Received these custom args: ['--checkpoint_name=model.ckpt-99', '--sample=8510']
('Run model here.', 'numactl --physcpubind=0-0 --membind=0 python generate.py /checkpoints/model.ckpt-99 --num_inter_threads=1 --num_intra_threads=1 --sample=8510')
PYTHONPATH: :/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py         --framework=tensorflow         --model-name=wavenet         --platform=fp32         --mode=inference         --model-source-dir=/workspace/models         --single-socket         --checkpoint=/checkpoints         --num-cores=1         --verbose         --checkpoint_name=model.ckpt-99         --sample=8510
Batch Size: -1
Ran inference with batch size -1
Log location outside container: /tmp/myuser/models/benchmarks/common/tensorflow/logs/benchmark_wavenet_inference.log
```
