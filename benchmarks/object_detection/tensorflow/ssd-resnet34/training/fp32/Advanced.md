<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# SSD-ResNet34 FP32 training - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running SSD-ResNet34 FP32
training, which provides more control over the individual parameters that
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
coco training dataset, TensorFlow models repo, and an output directory
where log files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
export DATASET_DIR=<path to the training dataset directory>
export OUTPUT_DIR=<directory where log files will be written>
```

Use the command below to run SSD-ResNet34 training. You can also edit parameters
like the `--num-train-steps` and `--mpi_num_processes` (to run multi-instance
training) as needed. If you want checkpoint files to be saved, specify the
`--checkpoint <directory path>` flag for the location where files will be written.

> Note: for best performance, use the same value for the arguments num-cores and num-intra-thread as follows:
>   For single instance run (mpi_num_processes=1): the value is equal to number of logical cores per socket.
>   For multi-instance run (mpi_num_processes > 1): the value is equal to (#_of_logical_cores_per_socket - 2).
>   If the `--num-cores` or `--num-intra-threads` args are not specified, these args will be calculated based on
>   the number of logical cores on your system.

```bash
python launch_benchmark.py \
    --data-location ${DATASET_DIR} \
    --model-source-dir ${TF_MODELS_DIR} \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --num-train-steps 100 \
    --num-cores 52 \
    --num-inter-threads 1 \
    --num-intra-threads 52 \
    --batch-size=100 \
    --weight_decay=1e-4 \
    --mpi_num_processes=1 \
    --mpi_num_processes_per_socket=1 \
    --output-dir ${OUTPUT_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest
```

