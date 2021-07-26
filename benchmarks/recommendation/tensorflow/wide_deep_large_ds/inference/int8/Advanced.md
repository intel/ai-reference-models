<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# Wide and Deep using a large dataset Int8 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running Wide and Deep using a large dataset Int8
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

export DATASET_DIR=<path to TF records file>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph .pb file>
```

Wide and Deep using a large dataset can be run in a few different modes:
* An accuracy test can be run using the command below.
  ```
  python launch_benchmark.py \
    --model-name wide_deep_large_ds \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1000 \
    --accuracy-only \
    --docker-image intel/intel-optimized-tensorflow:1.15.2 \
    --in-graph $PRETRAINED_MODEL \
    --data-location $DATASET_DIR \
    --output-dir $OUTPUT_DIR
  ```
* Run an online inference test to measure latency with `--batch-size 1`.
  ```
  python launch_benchmark.py \
    --model-name wide_deep_large_ds \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:1.15.2 \
    --in-graph $PRETRAINED_MODEL \
    --data-location $DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --num-intra-threads 1 --num-inter-threads 1  \
    -- num_omp_threads=1
  ```
* Run a batch inference test by setting `--batch-size 512`. The `numactl`
  command is a utility which can be used to control NUMA policy for processes
  or shared memory. To install numactl do:
  ```
  apt install numactl
  ```
  By default numactl is disabled. User can use NUMA policy control only on
  bare metal in advanced cases to specify number of cores to be used. This can
  be specified as shown in the command below only on bare metal. Below are the
  commands that gives best performance on 28 cores. The hyperparameters
  num-intra-threads, num-inter-threads, num_omp_threads etc should be tuned
  for best performance.
  * Case 1: Disabling `use_parallel_batches` option. In this case the batches
    are inferred in sequential order. By default `use_parallel_batches` is
    disabled. Kmp variables can also be set by using the arguments shown
    below or through config file
    `models/benchmarks/recommendation/tensorflow/wide_deep_large_ds/inference/int8/config.json`
    ```
    numactl --physcpubind=0-27 -m 0 python launch_benchmark.py \
        --model-name wide_deep_large_ds \
        --precision int8 \
        --mode inference \
        --framework tensorflow \
        --benchmark-only \
        --batch-size 512 \
        --in-graph $PRETRAINED_MODEL \
        --data-location $DATASET_DIR \
        --output-dir $OUTPUT_DIR \
        --num-intra-threads 28 --num-inter-threads 1 \
        -- num_omp_threads=16  kmp_block_time=1 kmp_settings=1
    ```
    The tail of the log output when the script completes should look something like this:
    ```
    --------------------------------------------------
    Total test records           :  2000000
    Batch size is                :  512
    Number of batches            :  3907
    Classification accuracy (%)  :  77.636
    Inference duration (seconds) :  3.1534
    Average Latency (ms/batch)   :  0.8285
    Throughput is (records/sec)  :  617992.762
    --------------------------------------------------
    ```
  * Case 2: Enabling `use_parallel_batches` option. In this case multiples
    batches are inferred in parallel. Number of batches to be executed in parallel
    can be given by argument num_parallel_batches.
    ```
    numactl --physcpubind=0-27 -m 0 python launch_benchmark.py \
        --model-name wide_deep_large_ds \
        --precision int8 \
        --mode inference \
        --framework tensorflow \
        --benchmark-only \
        --batch-size 512 \
        --in-graph $PRETRAINED_MODEL \
        --data-location $DATASET_DIR \
        --output-dir $OUTPUT_DIR \
        --num-intra-threads 1 --num-inter-threads 28  \
        -- num_omp_threads=1 use_parallel_batches=True num_parallel_batches=28 kmp_block_time=0 kmp_settings=1
    ```
    The tail of the log output when the script completes should look something like this:
    ```
    --------------------------------------------------
    Total test records           :  2000000
    Batch size is                :  512
    Number of batches            :  3907
    Inference duration (seconds) :  1.7056
    Average Latency (ms/batch)   :  12.2259
    Throughput is (records/sec)  :  1172597.21
    --------------------------------------------------
    ```

