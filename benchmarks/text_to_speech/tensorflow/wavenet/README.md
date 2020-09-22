# WaveNet

This document has instructions for how to run WaveNet for the following
modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model training and inference
for other precisions are coming later.

## FP32 Inference Instructions

1. Clone the [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
repo and get pull request #352 for the CPU optimizations.  The path to
the cloned repo will be passed as the model source directory when
running the launch script.

```
$ git clone https://github.com/ibab/tensorflow-wavenet.git
$ cd tensorflow-wavenet/

$ git fetch origin pull/352/head:cpu_optimized
$ git checkout cpu_optimized
```

2. Clone this [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, as well as
checkpoint files for a pre-trained model.  After cloning the repo,
navigate to the `benchmarks` directory, which is where the launch script
is located.

```
$ git clone https://github.com/IntelAI/models.git
$ cd models/benchmarks
```

3. Download and extract the pretrained model:
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/wavenet_fp32_pretrained_model.tar.gz
$ tar -xvf wavenet_fp32_pretrained_model.tar.gz
```

4. Start a model run by executing the launch script and passing args
specifying that we are running wavenet fp32 inference using TensorFlow,
along with a dockerfile that includes Intel Optimizations for TensorFlow
and the path to the model source dir (from step 1) and the checkpoint
files (from step 3).  We are also passing a couple of extra model args
for wavenet: the name of the checkpoint to use and the sample number.

```
python launch_benchmark.py \
    --precision fp32 \
    --model-name wavenet \
    --mode inference \
    --framework tensorflow \
    --socket-id 0 \
    --num-cores 1 \
    --docker-image intel/intel-optimized-tensorflow:1.15.2 \
    --model-source-dir /home/<user>/wavenet/tensorflow-wavenet \
    --checkpoint /home/<user>/wavenet_checkpoints \
    -- checkpoint_name=model.ckpt-99 sample=8510
```

5.  The logs are displayed in the console output as well as saved to a
file in the value of `--output-dir`.

The tail of the log should look something like this:

```
Time per 500 Samples: 1.530719 sec
Samples / sec: 326.643875
msec / sample: 3.061438
Sample: 7500
Time per 500 Samples: 1.553019 sec
Samples / sec: 321.953602
msec / sample: 3.106038
Sample: 8000
Time per 500 Samples: 1.552633 sec
Samples / sec: 322.033594
msec / sample: 3.105266
Sample: 8500

Average Throughput of whole run: Samples / sec: 289.351783
Average Latency of whole run: msec / sample: 3.456001
Finished generating. The result can be viewed in TensorBoard.
Ran inference with batch size -1
Log location outside container: {--output-dir value}/benchmark_wavenet_inference_fp32_20190105_015022.log
```
