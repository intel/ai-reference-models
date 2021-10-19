<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# Transformer Language BFloat16 training - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running Transformer Language BFloat16
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
the model zoo and set environment variables for the dataset, checkpoint
directory, and an output directory where log files will be written.
```
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files and checkpoints will be written>
```

Transformer Language in mlperf benchmark can run with full training or
fewer training steps. During training we can control if it will do the evaluation
or not.

Note user can specify the batch size suitable to their system with command option `--batch-size` to achieve the best performance. If users don't specify the batch size, then the model will choose default batch size, which is set to 5120.

For training with all epochs, with evaluation, and with saving checkpoints (for convergence):
```
python launch_benchmark.py \
    --framework tensorflow  \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    --socket-id 0 \
    --data-location ${DATASET_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    -- random_seed=11 train_steps=0 steps_between_eval=0 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50
```
The Tensorflow binary in the docker image needed to be compiled correctly so that Bfloat16 code is included.

For training with fewer training steps, such as 200 steps, and with no evaluation:
```
python launch_benchmark.py \
    --framework tensorflow \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    --socket-id 0 \
    --data-location ${DATASET_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="Yes" do_eval="No" print_iter=50
```

For training with fewer training steps, and with evaluation:
```
python launch_benchmark.py \
    --framework tensorflow \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    --socket-id 0 \
    --data-location ${DATASET_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50 \
    bleu_source=/home/<user>/newstest2014.en --bleu_ref=/home/<user>/newstest2014.de
```

For training only for Benchmarking, to reduce training time,
"saving checkpoints" and "doing the evaluation" can be disabled as below:
```
python launch_benchmark.py \
    --framework tensorflow \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    --socket-id 0 \
    --data-location ${DATASET_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="No" do_eval="No" print_iter=50
```

All the above commands run on a single socket of the CPU and calculate the number of cores
using `BaseModelInitializer.set_num_inter_intra_threads()`.

For training in multi-instance mode (2 sockets in a single node for example) in evaluation mode,
where you are "saving checkpoints" and "doing the evaluation":
```
python launch_benchmark.py \
    --framework tensorflow \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    --data-location ${DATASET_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --batch-size ${BATCH_SIZE} \
    --mpi_num_processes=2 \
    --num-intra-threads=26 --num-inter-threads=1 \
    -- random_seed=11 train_steps=0 steps_between_eval=0 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50
```

For training only in multi-instance mode (4 sockets in a single node for example) for benchmarking,
"saving checkpoints" and "doing the evaluation" can be disabled as below:
```
python launch_benchmark.py \
    --framework tensorflow \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    --data-location ${DATASET_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --batch-size ${BATCH_SIZE} \
    --mpi_num_processes=4 \
    --num-intra-threads=26 --num-inter-threads=1 \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="No" do_eval="No" print_iter=50
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output. The above two command assumes 2 sockets in the CPU,
and 28 cores per socket. We keep aside 2 cores for horovod communications. Thus,
in general the num-intra-threads is equal to "available cores per socket - 2".

The transformer model related parameters is appended after "-- "
the num-intra-threads is equal to available cores - 2.

The log file is saved to the `$OUTPUT_DIR` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running training without saving checkpoints and not doing the evaluations:
```
INFO:tensorflow:loss = 9.804764, step = 0
I0430 15:43:41.658226 140067518572352 basic_session_run_hooks.py:262] loss = 9.804764, step = 0
INFO:tensorflow:loss = 9.804764
I0430 15:43:41.658570 140067518572352 basic_session_run_hooks.py:262] loss = 9.804764
INFO:tensorflow:Batch [10]:  last 10 steps exp/sec = 1777.54, completed 10/50 warmup steps
I0430 15:44:10.461924 140067518572352 hooks.py:117] Batch [10]:  last 10 steps exp/sec = 1777.54, completed 10/50 warmup steps
INFO:tensorflow:loss = 9.741728 (28.804 sec)
I0430 15:44:10.462378 140067518572352 basic_session_run_hooks.py:260] loss = 9.741728 (28.804 sec)
INFO:tensorflow:Batch [20]:  last 10 steps exp/sec = 1910.88, completed 20/50 warmup steps
I0430 15:44:37.255764 140067518572352 hooks.py:117] Batch [20]:  last 10 steps exp/sec = 1910.88, completed 20/50 warmup steps
INFO:tensorflow:loss = 9.638916 (26.794 sec)
I0430 15:44:37.255995 140067518572352 basic_session_run_hooks.py:260] loss = 9.638916 (26.794 sec)
INFO:tensorflow:Batch [30]:  last 10 steps exp/sec = 1898.28, completed 30/50 warmup steps
I0430 15:45:04.227502 140067518572352 hooks.py:117] Batch [30]:  last 10 steps exp/sec = 1898.28, completed 30/50 warmup steps
INFO:tensorflow:loss = 9.4281845 (26.972 sec)
I0430 15:45:04.227731 140067518572352 basic_session_run_hooks.py:260] loss = 9.4281845 (26.972 sec)
INFO:tensorflow:Batch [40]:  last 10 steps exp/sec = 1869.21, completed 40/50 warmup steps
I0430 15:45:31.618821 140067518572352 hooks.py:117] Batch [40]:  last 10 steps exp/sec = 1869.21, completed 40/50 warmup steps
INFO:tensorflow:loss = 9.169721 (27.392 sec)
I0430 15:45:31.619261 140067518572352 basic_session_run_hooks.py:260] loss = 9.169721 (27.392 sec)
INFO:tensorflow:Batch [50]:  last 10 steps exp/sec = 1943.57, completed 50/50 warmup steps
I0430 15:45:57.962067 140067518572352 hooks.py:117] Batch [50]:  last 10 steps exp/sec = 1943.57, completed 50/50 warmup steps
```

Multi-instance sample log:
Running 2 instances only. Both instances printing loss values.
```
INFO:tensorflow:loss = 9.881177, step = 0
I0430 14:29:19.457713 140147178080064 basic_session_run_hooks.py:262] loss = 9.881177, step = 0
INFO:tensorflow:loss = 9.881177
I0430 14:29:19.458147 140147178080064 basic_session_run_hooks.py:262] loss = 9.881177
INFO:tensorflow:loss = 9.916411, step = 0
I0430 14:29:19.486985 140684161857344 basic_session_run_hooks.py:262] loss = 9.916411, step = 0
INFO:tensorflow:loss = 9.916411
I0430 14:29:19.487454 140684161857344 basic_session_run_hooks.py:262] loss = 9.916411
INFO:tensorflow:Batch [10]:  last 10 steps exp/sec = 1368.86, completed 10/50 warmup steps
I0430 14:29:56.861132 140147178080064 hooks.py:117] Batch [10]:  last 10 steps exp/sec = 1368.86, completed 10/50 warmup steps
INFO:tensorflow:loss = 9.821904 (37.403 sec)
I0430 14:29:56.861575 140147178080064 basic_session_run_hooks.py:260] loss = 9.821904 (37.403 sec)
INFO:tensorflow:Batch [10]:  last 10 steps exp/sec = 1369.8, completed 10/50 warmup steps
I0430 14:29:56.864749 140684161857344 hooks.py:117] Batch [10]:  last 10 steps exp/sec = 1369.8, completed 10/50 warmup steps
INFO:tensorflow:loss = 9.83847 (37.378 sec)
I0430 14:29:56.865195 140684161857344 basic_session_run_hooks.py:260] loss = 9.83847 (37.378 sec)
INFO:tensorflow:Batch [20]:  last 10 steps exp/sec = 1364.68, completed 20/50 warmup steps
I0430 14:30:34.378897 140147178080064 hooks.py:117] Batch [20]:  last 10 steps exp/sec = 1364.68, completed 20/50 warmup steps
INFO:tensorflow:loss = 9.549045 (37.518 sec)
```

