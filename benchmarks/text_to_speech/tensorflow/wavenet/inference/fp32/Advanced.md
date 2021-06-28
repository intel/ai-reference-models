<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# WaveNet FP32 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running WaveNet FP32
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
dataset, pretrained model, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model checkpoints>
export TF_WAVENET_DIR=<path to your clone of the TensorFlow WaveNet repo>
```

Start a model run by executing the launch script and passing args
specifying that we are running wavenet fp32 inference using TensorFlow,
along with a docker image that includes Intel Optimizations for TensorFlow
and the path to your clone of the WaveNet repo and the checkpoint
files that were downloaded.  We are also passing a couple of extra model args
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
    --model-source-dir ${TF_WAVENET_DIR} \
    --checkpoint ${PRETRAINED_MODEL} \
    --output-dir ${OUTPUT_DIR} \
    -- checkpoint_name=model.ckpt-99 sample=8510
```

The logs are displayed in the console output as well as saved to a
file in the value of `${OUTPUT_DIR}`.

The tail of the log should look something like this:
```
Time per 500 Samples: ... sec
Samples / sec: ...
msec / sample: ...
Sample: 8000
Time per 500 Samples: ... sec
Samples / sec: ...
msec / sample: ...
Sample: 8500

Average Throughput of whole run: Samples / sec: ...
Average Latency of whole run: msec / sample: ...
Finished generating. The result can be viewed in TensorBoard.
Log file location: ${OUTPUT_DIR}/benchmark_wavenet_inference_fp32_20210601_143852.log
```

