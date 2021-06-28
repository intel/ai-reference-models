<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# 3D U-Net FP32 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running 3D U-Net FP32
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

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model>
```

3D U-Net FP32 inference can be run with a batch size of 1:
```
python launch_benchmark.py \
  --precision fp32 \
  --model-name 3d_unet \
  --mode inference \
  --framework tensorflow \
  --docker-image intel/intel-optimized-tensorflow:1.15.2 \
  --in-graph $PRETRAINED_MODEL \
  --data-location ${DATASET_DIR} \
  --batch-size 1 \
  --socket-id 0 \
  --output-dir ${OUTPUT_DIR} 
```

Below is an example tail of the log file:

```
Loading pre-trained model
Time spent per BATCH: ... ms
Total samples/sec: ... samples/s
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_3d_unet_inference_fp32_20190116_234659.log
```

