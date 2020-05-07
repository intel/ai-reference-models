# Transformer Language

This document has instructions for how to run Transformer Language used in mlperf
Benchmark suits for the following modes/platforms:
* [FP32 training](#fp32-training-instructions)
* [Bfloat16 training](#bfloat16-training-instructions)

Detailed information on Benchmark can be found in [mlperf/training](https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)

Instructions and scripts for model training and inference for
other platforms are coming later.

## FP32 Training Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

2. Obtain the dataset.
Decide the problem you want to run to get the appropriate dataset.
We will get the training data of it as an example:

    Download dataset for computing BLEU score reported in the paper
    ```
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
    ```
    Download training dataset
    ```
    $ export PYTHONPATH=$PYTHONPATH:/home/<user>/models/models/common/tensorflow
    $ export DATA_DIR=/home/<user>/transformer_data
    $ cd /home/<user>/models/models/language_translation/tensorflow/transformer_mlperf/training/bfloat16/transformer
    $ python data_download.py --data_dir=$DATA_DIR
    ```
    
3. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo (from step 1).
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the dataset location (from step 2).

Transformer Language in mlperf benchmark can run with full training or
fewer training steps. During training we can control if it will do the evaluation
or not.  

For training with all epochs, with evaluation, and with saving checkpoints (for convergence):

```
python launch_benchmark.py \
    --framework tensorflow \ 
    --precision fp32 \
    --mode training \
    --model-name transformer_mlperf \
    -i 0 --data-location $DATA_DIR \
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=28 --num-inter-threads=1 \
    -- random_seed=11 train_steps=0 steps_between_eval=0 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50
```
For training with fewer training steps, and with no evaluation:

```
python launch_benchmark.py \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --model-name transformer_mlperf \
    -i 0 --data-location $DATA_DIR \ 
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=28 --num-inter-threads=1 \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="Yes" do_eval="No" print_iter=50
```

For training with fewer training steps, and with evaluation:

```
python launch_benchmark.py \
    --framework tensorflow \ 
    --precision fp32 \ 
    --mode training \
    --model-name transformer_mlperf \
    -i 0 --data-location $DATA_DIR \
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=28 --num-inter-threads=1 \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50 \
    bleu_source=/home/<user>/newstest2014.en --bleu_ref=/home/<user>/newstest2014.de
```

For training only for Benchmarking, to reduce training time,
"saving checkpoints" and "doing the evaluation" can be disabled as below:

```
python launch_benchmark.py \
    --framework tensorflow \ 
    --precision fp32 \ 
    --mode training \
    --model-name transformer_mlperf \
    -i 0 --data-location $DATA_DIR \
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=28 --num-inter-threads=1 \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="No" do_eval="No" print_iter=50
```
All the above command runs on a single socket of the CPU, where each socket has 28 cores. In general,
please use --num-intra-threads="number of cores per socket" for the best threading performance. 

For training in multi-instance mode (2 sockets in a single node for example) in evaluation mode,
where we are "saving checkpoints" and "doing the evaluation":

```
python launch_benchmark.py \ 
    --framework tensorflow \ 
    --precision fp32 \ 
    --mode training \
    --model-name transformer_mlperf \
    --data-location $DATA_DIR \
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=26 --num-inter-threads=1 \
    --mpi_num_processes=2 \
    -- random_seed=11 train_steps=0 steps_between_eval=0 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50
```

For training only in multi-instance mode (4 sockets in a single node for example) for benchmrking,
"saving checkpoints" and "doing the evaluation" can be disabled as below:

```
python launch_benchmark.py \ 
    --framework tensorflow \ 
    --precision fp32 \ 
    --mode training \
    --model-name transformer_mlperf \
    --data-location $DATA_DIR \
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=26 --num-inter-threads=1 \
    --mpi_num_processes=2 \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="No" do_eval="No" print_iter=50
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output. The above two command assumes 2 sockets in the CPU,
and 28 cores per socket. We keep aside 2 cores for horovod communications. Thus,
in general the num-intra-threads is equal to "available cores per socket - 2". 
The transformer model related parameters is appended after "-- "

4.  The log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Single Instance sample log: 
Example log tail when running training with saving checkpoints, but not doing the evaluations:
```
I1010 00:14:10.006996 140608628639552 monitored_session.py:240] Graph was finalized.
2019-10-10 00:14:10.007365: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2019-10-10 00:14:10.045231: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2500000000 Hz
2019-10-10 00:14:10.050200: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0xc7227f0 executing computations on platform Host. Devices:
2019-10-10 00:14:10.050220: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
INFO:tensorflow:Running local_init_op.
I1010 00:14:15.688582 140608628639552 session_manager.py:500] Running local_init_op.
INFO:tensorflow:Done running local_init_op.
I1010 00:14:15.954741 140608628639552 session_manager.py:502] Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into /tmp/transformer_model/model.ckpt.
I1010 00:14:21.952126 140608628639552 basic_session_run_hooks.py:606] Saving checkpoints for 0 into /tmp/transformer_model/model.ckpt.
:::MLPv0.5.0 [1df9b2ab-3dc8-478b-8682-ce574b800024][1570666471.29151][0]
INFO:tensorflow:loss = 9.863386, step = 0
I1010 00:14:35.181882 140608628639552 basic_session_run_hooks.py:262] loss = 9.863386, step = 0
INFO:tensorflow:Saving checkpoints for 5 into /tmp/transformer_model/model.ckpt.
I1010 00:15:02.218935 140608628639552 basic_session_run_hooks.py:606] Saving checkpoints for 5 into /tmp/transformer_model/model.ckpt.
INFO:tensorflow:Loss for final step: 9.856145.
I1010 00:15:05.596456 140608628639552 estimator.py:371] Loss for final step: 9.856145.
:::MLPv0.5.0 transformer 1570666505.598165274 (transformer/transformer_main.py:312) eval_start
I1010 00:15:05.598198 140608628639552 mlperf_log.py:134] :::MLPv0.5.0 transformer ... (transformer/transformer_main.py:312) eval_start
:::MLPv0.5.0 transformer 1570666505.598524094 (transformer/transformer_main.py:328) eval_stop
I1010 00:15:05.598533 140608628639552 mlperf_log.py:134] :::MLPv0.5.0 transformer ... (transformer/transformer_main.py:328) eval_stop
:::MLPv0.5.0 transformer 1570666505.598799944 (transformer/transformer_main.py:399) run_stop
I1010 00:15:05.598808 140608628639552 mlperf_log.py:134] :::MLPv0.5.0 transformer ... (transformer/transformer_main.py:399) run_stop
:::MLPv0.5.0 transformer 1570666505.599067211 (transformer/transformer_main.py:400) run_final
I1010 00:15:05.599075 140608628639552 mlperf_log.py:134] :::MLPv0.5.0 transformer ... (transformer/transformer_main.py:400) run_final
lscpu_path_cmd = command -v lscpu
lscpu located here: b'/usr/bin/lscpu'
num_inter_threads: 2
num_intra_threads: 56

````

Multi-instance sample log:
Running 2 instances only. Both the instance prining loss values.

```
Example log tail when running training with saving checkpoints, but not doing the evaluations:
I0408 14:32:44.900605 140564697147200 hooks.py:117] Batch [10]:  last 10 steps exp/sec = TBD, completed 10/20 wamrup steps
I0408 14:32:44.900944 139850962708288 basic_session_run_hooks.py:260] loss = 10.014245 (TBD sec)
INFO:tensorflow:loss = 9.9818325 (... sec)
I0408 14:32:44.901092 140564697147200 basic_session_run_hooks.py:260] loss = 9.9818325 (TBD sec)
INFO:tensorflow:Batch [20]:  last 10 steps exp/sec = 1021.16, completed 20/20 wamrup steps
I0408 14:33:25.011524 139850962708288 hooks.py:117] Batch [20]:  last 10 steps exp/sec = TBD, completed 20/20 wamrup steps
INFO:tensorflow:loss = 9.758567 (... sec)
I0408 14:33:25.012046 139850962708288 basic_session_run_hooks.py:260] loss = 9.758567 (TBD sec)
INFO:tensorflow:Batch [20]:  last 10 steps exp/sec = 1021.11, completed 20/20 wamrup steps
I0408 14:33:25.013776 140564697147200 hooks.py:117] Batch [20]:  last 10 steps exp/sec = TBD, completed 20/20 wamrup steps
INFO:tensorflow:loss = 9.7565155 (... sec)
I0408 14:33:25.014259 140564697147200 basic_session_run_hooks.py:260] loss = 9.7565155 (TBD sec)
INFO:tensorflow:Batch [30]:  last 10 steps exp/sec = 1011.26, total average exp/sec = TBD
I0408 14:34:05.515469 139850962708288 hooks.py:113] Batch [30]:  last 10 steps exp/sec = TBD, total average exp/sec = TBD
INFO:tensorflow:loss = 9.43642 (... sec)
I0408 14:34:05.515967 139850962708288 basic_session_run_hooks.py:260] loss = 9.43642 (TBD sec)
INFO:tensorflow:Batch [30]:  last 10 steps exp/sec = 1011.31, total average exp/sec = TBD
I0408 14:34:05.515844 140564697147200 hooks.py:113] Batch [30]:  last 10 steps exp/sec = TBD, total average exp/sec = TBD
INFO:tensorflow:loss = 9.385353 (... sec)
I0408 14:34:05.516412 140564697147200 basic_session_run_hooks.py:260] loss = 9.385353 (... sec)
```

## Bfloat16 Training Instructions
(Experimental)

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

2. Obtain the dataset
Decide the problem you want to run to get the appropriate dataset.
We will get the training data of it as an example:

    Download dataset for computing BLEU score reported in the paper
    ```
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
    ```
    Download training dataset
    ```
    $ export PYTHONPATH=$PYTHONPATH:/home/<user>/models/models/common/tensorflow
    $ export DATA_DIR=/home/<user>/transformer_data
    $ cd /home/<user>/models/models/language_translation/tensorflow/transformer_mlperf/training/bfloat16/transformer
    $ python data_download.py --data_dir=$DATA_DIR
    ```

3. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo (from step 1).
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the dataset location (from step 2).

Transformer Language in mlperf benchmark can run with full training or
fewer training steps. During training we can control if it will do the evaluation
or not.

For training with all epochs, with evaluation, and with saving checkpoints (for convergence):

```
python launch_benchmark.py \
    --framework tensorflow  \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    -i 0 --data-location $DATA_DIR \ 
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=28 --num-inter-threads=1 \
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
    -i 0 --data-location $DATA_DIR \ 
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=28 --num-inter-threads=1 \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="Yes" do_eval="No" print_iter=50
```

For training with fewer training steps, and with evaluation:

```
python launch_benchmark.py \
    --framework tensorflow \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    -i 0 --data-location $DATA_DIR \ 
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=28 --num-inter-threads=1 \
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
    -i 0 --data-location $DATA_DIR \ 
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --num-intra-threads=28 --num-inter-threads=1 \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="No" do_eval="No" print_iter=50
```
All the above command runs on a single socket of the CPU, where each socket has 28 cores. In general,
please use --num-intra-threads="number of cores per socket" for the best threading performance. 

For training in multi-instance mode (2 sockets in a single node for example) in evaluation mode,
where you are "saving checkpoints" and "doing the evaluation":

```
python launch_benchmark.py \ 
    --framework tensorflow \ 
    --precision bfloat16 \ 
    --mode training \
    --model-name transformer_mlperf \
    --data-location $DATA_DIR \ 
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
    --mpi_num_processes=2 \
    --num-intra-threads=26 --num-inter-threads=1 \
    -- random_seed=11 train_steps=0 steps_between_eval=0 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50
```
For training only in multi-instance mode (4 sockets in a single node for example) for benchmrking,
"saving checkpoints" and "doing the evaluation" can be disabled as below:

```
python launch_benchmark.py \ 
    --framework tensorflow \ 
    --precision bfloat16 \ 
    --mode training \
    --model-name transformer_mlperf \
    --data-location $DATA_DIR \ 
    --docker-image intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
    --verbose \
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
The transformer model related parameters is appended after "-- "

4.  The log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running training with saving checkpoints, but not doing the evaluations:
```
I1010 00:14:10.006996 140608628639552 monitored_session.py:240] Graph was finalized.
2019-10-10 00:14:10.007365: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2019-10-10 00:14:10.045231: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2500000000 Hz
2019-10-10 00:14:10.050200: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0xc7227f0 executing computations on platform Host. Devices:
2019-10-10 00:14:10.050220: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
INFO:tensorflow:Running local_init_op.
I1010 00:14:15.688582 140608628639552 session_manager.py:500] Running local_init_op.
INFO:tensorflow:Done running local_init_op.
I1010 00:14:15.954741 140608628639552 session_manager.py:502] Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into /tmp/transformer_model/model.ckpt.
I1010 00:14:21.952126 140608628639552 basic_session_run_hooks.py:606] Saving checkpoints for 0 into /tmp/transformer_model/model.ckpt.
:::MLPv0.5.0 [1df9b2ab-3dc8-478b-8682-ce574b800024][1570666471.29151][0]
INFO:tensorflow:loss = 9.863386, step = 0
I1010 00:14:35.181882 140608628639552 basic_session_run_hooks.py:262] loss = 9.863386, step = 0
INFO:tensorflow:Saving checkpoints for 5 into /tmp/transformer_model/model.ckpt.
I1010 00:15:02.218935 140608628639552 basic_session_run_hooks.py:606] Saving checkpoints for 5 into /tmp/transformer_model/model.ckpt.
INFO:tensorflow:Loss for final step: 9.856145.
I1010 00:15:05.596456 140608628639552 estimator.py:371] Loss for final step: 9.856145.
:::MLPv0.5.0 transformer 1570666505.598165274 (transformer/transformer_main.py:312) eval_start
I1010 00:15:05.598198 140608628639552 mlperf_log.py:134] :::MLPv0.5.0 transformer ... (transformer/transformer_main.py:312) eval_start
:::MLPv0.5.0 transformer 1570666505.598524094 (transformer/transformer_main.py:328) eval_stop
I1010 00:15:05.598533 140608628639552 mlperf_log.py:134] :::MLPv0.5.0 transformer ... (transformer/transformer_main.py:328) eval_stop
:::MLPv0.5.0 transformer 1570666505.598799944 (transformer/transformer_main.py:399) run_stop
I1010 00:15:05.598808 140608628639552 mlperf_log.py:134] :::MLPv0.5.0 transformer ... (transformer/transformer_main.py:399) run_stop
:::MLPv0.5.0 transformer 1570666505.599067211 (transformer/transformer_main.py:400) run_final
I1010 00:15:05.599075 140608628639552 mlperf_log.py:134] :::MLPv0.5.0 transformer ... (transformer/transformer_main.py:400) run_final
lscpu_path_cmd = command -v lscpu
lscpu located here: b'/usr/bin/lscpu'
num_inter_threads: 2
num_intra_threads: 56
```

Multi-instance sample log:
Running 2 instances only. Both the instance prining loss values.

```
I0408 14:32:44.900605 140564697147200 hooks.py:117] Batch [10]:  last 10 steps exp/sec = TBD, completed 10/20 wamrup steps
I0408 14:32:44.900944 139850962708288 basic_session_run_hooks.py:260] loss = 10.014245 (TBD sec)
INFO:tensorflow:loss = 9.9818325 (... sec)
I0408 14:32:44.901092 140564697147200 basic_session_run_hooks.py:260] loss = 9.9818325 (TBD sec)
INFO:tensorflow:Batch [20]:  last 10 steps exp/sec = 1021.16, completed 20/20 wamrup steps
I0408 14:33:25.011524 139850962708288 hooks.py:117] Batch [20]:  last 10 steps exp/sec = TBD, completed 20/20 wamrup steps
INFO:tensorflow:loss = 9.758567 (... sec)
I0408 14:33:25.012046 139850962708288 basic_session_run_hooks.py:260] loss = 9.758567 (TBD sec)
INFO:tensorflow:Batch [20]:  last 10 steps exp/sec = 1021.11, completed 20/20 wamrup steps
I0408 14:33:25.013776 140564697147200 hooks.py:117] Batch [20]:  last 10 steps exp/sec = TBF, completed 20/20 wamrup steps
INFO:tensorflow:loss = 9.7565155 (... sec)
I0408 14:33:25.014259 140564697147200 basic_session_run_hooks.py:260] loss = 9.7565155 (TBD sec)
INFO:tensorflow:Batch [30]:  last 10 steps exp/sec = 1011.26, total average exp/sec = TBD
I0408 14:34:05.515469 139850962708288 hooks.py:113] Batch [30]:  last 10 steps exp/sec = TBD, total average exp/sec = ...
INFO:tensorflow:loss = 9.43642 (... sec)
I0408 14:34:05.515967 139850962708288 basic_session_run_hooks.py:260] loss = 9.43642 (TBD sec)
INFO:tensorflow:Batch [30]:  last 10 steps exp/sec = 1011.31, total average exp/sec = TBD
I0408 14:34:05.515844 140564697147200 hooks.py:113] Batch [30]:  last 10 steps exp/sec = TBD total average exp/sec = TBD
INFO:tensorflow:loss = 9.385353 (... sec)
I0408 14:34:05.516412 140564697147200 basic_session_run_hooks.py:260] loss = 9.385353 (TBD)
```
