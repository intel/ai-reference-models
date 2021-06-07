# Transformer Language

This document has instructions for how to run Transformer Language used in mlperf
Benchmark suits for the following modes/platforms:
* [FP32 training](#fp32-training-instructions)
* [Bfloat16 training](#bfloat16-training-instructions)
* [FP32 inference](#fp32-inference-instructions)
* [Bfloat16 inference](#bfloat16-inference-instructions)

Detailed information on Benchmark can be found in [mlperf/training](https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)

Instructions and scripts for model training and inference for
other platforms are coming later.

# <a name="fp32-training-instructions"></a> FP32 Training Instructions

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
    $ export DATA_DIR=/home/<user>/transformer_data
    $ mkdir $DATA_DIR && cd $DATA_DIR
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
    ```
    Download<sup id="a1">[1](#f1)</sup> the training dataset<sup id="a2">[2](#f2)</sup>. 
    ```
    $ export PYTHONPATH=$PYTHONPATH:/home/<user>/models/models/common/tensorflow
    $ export DATA_DIR=/home/<user>/transformer_data
    $ cd /home/<user>/models/models/language_translation/tensorflow/transformer_mlperf/training/fp32/transformer
    $ python data_download.py --data_dir=$DATA_DIR
    ```

---

<sub><b id="f1">1 - </b> Running `python data_download.py --data_dir=$DATA_DIR` assumes you have a python environment similar to what the intel/intel-optimized-tensorflow:2.4.0 container provides. One option would be to run the above within the intel/intel-optimized-tensorflow:2.4.0 container eg: `docker run -u $(id -u):$(id -g) --privileged  --entrypoint /bin/bash -v /home/<user>:/home/<user> -it intel/intel-optimized-tensorflow:latest` [↩](#a1)</sub><br/>
<sub><b id="f2">2 - </b> Downloading the datasets can take some time, you should see `XX% completed` updates for the datasets [↩](#a2)</sub>

---
    
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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="No" do_eval="No" print_iter=50
```

All the above commands run on a single socket of the CPU. The calculation of the number of cores uses [BaseModelInitializer.set_num_inter_intra_threads()](/benchmarks/common/base_model_init.py#L152).

For training in multi-instance mode (2 sockets in a single node for example) in evaluation mode,
where we are "saving checkpoints" and "doing the evaluation":

```
python launch_benchmark.py \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --model-name transformer_mlperf \
    --data-location $DATA_DIR \
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
    --num-intra-threads=26 --num-inter-threads=1 \
    --mpi_num_processes=2 \
    -- random_seed=11 train_steps=0 steps_between_eval=0 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50
```

For training only in multi-instance mode (4 sockets in a single node for example) for benchmarking,
"saving checkpoints" and "doing the evaluation" can be disabled as below:

```
python launch_benchmark.py \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --model-name transformer_mlperf \
    --data-location $DATA_DIR \
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
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
Example log tail when running training without saving checkpoints and not doing the evaluations:
```
I0820 18:29:53.357862 140273402423104 basic_session_run_hooks.py:260] loss = 7.9346876 (190.364 sec)
INFO:tensorflow:Loss for final step: 7.4975414.
I0820 18:33:01.786211 140273402423104 estimator.py:350] Loss for final step: 7.4975414.
:::MLPv0.5.0 transformer 1597948381.787289143 (transformer/transformer_main.py:351) eval_start
I0820 18:33:01.787313 140273402423104 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597948381.787289143 (transformer/transformer_main.py:351) eval_start
:::MLPv0.5.0 transformer 1597948381.787651062 (transformer/transformer_main.py:367) eval_stop
I0820 18:33:01.787661 140273402423104 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597948381.787651062 (transformer/transformer_main.py:367) eval_stop
:::MLPv0.5.0 transformer 1597948381.787932158 (transformer/transformer_main.py:451) run_stop
I0820 18:33:01.787940 140273402423104 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597948381.787932158 (transformer/transformer_main.py:451) run_stop
:::MLPv0.5.0 transformer 1597948381.788205624 (transformer/transformer_main.py:452) run_final
I0820 18:33:01.788213 140273402423104 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597948381.788205624 (transformer/transformer_main.py:452) run_final
num_inter_threads: 1
num_intra_threads: 28
Received these standard args: Namespace(accuracy_only=False, backbone_model=None, batch_size=-1, benchmark_dir='/workspace/benchmarks', benchmark_only=True, bleu_ref=None, bleu_source=None, bleu_threshold=None, checkpoint=None, data_location='/dataset', data_num_inter_threads=None, data_num_intra_threads=None, disable_tcmalloc=True, do_eval='No', epochs_between_eval=1, experimental_gelu=False, framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', learning_rate=2, mode='training', model_args=[], model_name='transformer_mlperf', model_source_dir='/workspace/models', mpi=None, mpi_hostnames=None, num_cores=-1, num_cpu_cores=4, num_instances=1, num_inter_threads=1, num_intra_threads=28, num_mpi=1, num_train_steps=1, optimized_softmax=True, output_dir='/workspace/benchmarks/common/tensorflow/logs', output_results=False, params='big', precision='fp32', print_iter=50, random_seed=11, save_checkpoints='No', save_profile='', socket_id=0, static_batch='No', steps_between_eval=200, tcmalloc_large_alloc_report_threshold=2147483648, tf_serving_version='master', train_epochs=None, train_steps=200, use_case='language_translation', verbose=True)
Received these custom args: ['--random_seed=11', '--params=big', '--train_steps=200', '--steps_between_eval=200', '--do_eval=No', '--save_checkpoints=No', '--print_iter=50', '--save_profile=']
Current directory: /workspace/benchmarks
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/training/fp32/transformer/transformer_main.py --data_dir=/dataset --random_seed=11 --params=big --train_steps=200 --steps_between_eval=200 --do_eval=No --save_checkpoints=No --save_profile= --print_iter=50 --inter_op_parallelism_threads=1 --intra_op_parallelism_threads=28 --learning_rate=2 --static_batch=No
PYTHONPATH: :/workspace/intelai_models_common:/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=language_translation --model-name=transformer_mlperf --precision=fp32 --mode=training --benchmark-dir=/workspace/benchmarks --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=-1 --socket-id=0 --output-dir=/workspace/benchmarks/common/tensorflow/logs --num-train-steps=1  --benchmark-only  --verbose --model-source-dir=/workspace/models --data-location=/dataset --disable-tcmalloc=True --random_seed=11 --params=big --train_steps=200 --steps_between_eval=200 --do_eval=No --save_checkpoints=No
    --print_iter=50 --save_profile=
Log file location: /tmp/models/benchmarks/common/tensorflow/logs/benchmark_transformer_mlperf_training_fp32_20200820_181929.log
```

Multi-instance sample log:
Running 2 instances only. Both instances printing loss values.

```
I0820 20:15:17.018405 140439206057792 basic_session_run_hooks.py:260] loss = 7.363133 (230.673 sec)
INFO:tensorflow:Loss for final step: 7.381052.
I0820 20:19:05.257017 140439206057792 estimator.py:350] Loss for final step: 7.381052.
:::MLPv0.5.0 transformer 1597954745.258936405 (transformer/transformer_main.py:351) eval_start
I0820 20:19:05.258977 140439206057792 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597954745.258936405 (transformer/transformer_main.py:351) eval_start
:::MLPv0.5.0 transformer 1597954745.259335518 (transformer/transformer_main.py:367) eval_stop
I0820 20:19:05.259346 140439206057792 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597954745.259335518 (transformer/transformer_main.py:367) eval_stop
:::MLPv0.5.0 transformer 1597954745.259663582 (transformer/transformer_main.py:451) run_stop
I0820 20:19:05.259672 140439206057792 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597954745.259663582 (transformer/transformer_main.py:451) run_stop
:::MLPv0.5.0 transformer 1597954745.259935141 (transformer/transformer_main.py:452) run_final
I0820 20:19:05.259943 140439206057792 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597954745.259935141 (transformer/transformer_main.py:452) run_final
INFO:tensorflow:Loss for final step: 7.128895.
I0820 20:19:05.259997 140209294473024 estimator.py:350] Loss for final step: 7.128895.
:::MLPv0.5.0 transformer 1597954745.261870861 (transformer/transformer_main.py:351) eval_start
I0820 20:19:05.261909 140209294473024 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597954745.261870861 (transformer/transformer_main.py:351) eval_start
:::MLPv0.5.0 transformer 1597954745.262327194 (transformer/transformer_main.py:367) eval_stop
I0820 20:19:05.262341 140209294473024 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597954745.262327194 (transformer/transformer_main.py:367) eval_stop
:::MLPv0.5.0 transformer 1597954745.262665272 (transformer/transformer_main.py:451) run_stop
I0820 20:19:05.262675 140209294473024 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597954745.262665272 (transformer/transformer_main.py:451) run_stop
:::MLPv0.5.0 transformer 1597954745.262959242 (transformer/transformer_main.py:452) run_final
I0820 20:19:05.262968 140209294473024 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597954745.262959242 (transformer/transformer_main.py:452) run_final
num_inter_threads: 1
num_intra_threads: 28
Received these standard args: Namespace(accuracy_only=False, backbone_model=None, batch_size=-1, benchmark_dir='/workspace/benchmarks', benchmark_only=True, bleu_ref=None, bleu_source=None, bleu_threshold=None, checkpoint=None, data_location='/dataset', data_num_inter_threads=None, data_num_intra_threads=None, disable_tcmalloc=True, do_eval='No', epochs_between_eval=1, experimental_gelu=False, framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', learning_rate=2, mode='training', model_args=[], model_name='transformer_mlperf', model_source_dir='/workspace/models', mpi=None, mpi_hostnames=None, num_cores=-1, num_cpu_cores=4, num_instances=1, num_inter_threads=1, num_intra_threads=28, num_mpi=1, num_train_steps=1, optimized_softmax=True, output_dir='/workspace/benchmarks/common/tensorflow/logs', output_results=False, params='big', precision='fp32', print_iter=50, random_seed=11, save_checkpoints='No', save_profile='', socket_id=0, static_batch='No', steps_between_eval=200, tcmalloc_large_alloc_report_threshold=2147483648, tf_serving_version='master', train_epochs=None, train_steps=200, use_case='language_translation', verbose=True)
Received these custom args: ['--random_seed=11', '--params=big', '--train_steps=200', '--steps_between_eval=200', '--do_eval=No', '--save_checkpoints=No', '--print_iter=50', '--save_profile=']
Current directory: /workspace/benchmarks
Running: numactl --cpunodebind=0 --membind=0 mpirun --allow-run-as-root -n 2 --map-by socket python /workspace/intelai_models/training/fp32/transformer/transformer_main.py --data_dir=/dataset --random_seed=11 --params=big --train_steps=200 --steps_between_eval=200 --do_eval=No --save_checkpoints=No --save_profile= --print_iter=50 --inter_op_parallelism_threads=1 --intra_op_parallelism_threads=28 --learning_rate=2 --static_batch=No
PYTHONPATH: :/workspace/intelai_models_common:/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=language_translation --model-name=transformer_mlperf --precision=fp32 --mode=training --benchmark-dir=/workspace/benchmarks --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=-1 --socket-id=0 --output-dir=/workspace/benchmarks/common/tensorflow/logs --num-train-steps=1  --benchmark-only  --verbose --model-source-dir=/workspace/models --data-location=/dataset --disable-tcmalloc=True --random_seed=11 --params=big --train_steps=200 --steps_between_eval=200 --do_eval=No --save_checkpoints=No
    --print_iter=50 --save_profile=
Log file location: /tmp/models/benchmarks/common/tensorflow/logs/benchmark_transformer_mlperf_training_fp32_20200820_200237.log
```

## <a name="bfloat16-training-instructions"></a> Bfloat16 Training Instructions

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
    Download<sup id="a3">[3](#f3)</sup> the training dataset<sup id="a4">[4](#f4)</sup>. 
    ```
    $ export PYTHONPATH=$PYTHONPATH:/home/<user>/models/models/common/tensorflow
    $ export DATA_DIR=/home/<user>/transformer_data
    $ cd /home/<user>/models/models/language_translation/tensorflow/transformer_mlperf/training/bfloat16/transformer
    $ python data_download.py --data_dir=$DATA_DIR
    ```

---

<sub><b id="f3">3 - </b> Running `python data_download.py --data_dir=$DATA_DIR` assumes you have a python environment similar to what the intel/intel-optimized-tensorflow:2.4.0 container provides. One option would be to run the above within the intel/intel-optimized-tensorflow:2.4.0 container eg: `docker run -u $(id -u):$(id -g) --privileged  --entrypoint /bin/bash -v /home/<user>:/home/<user> -it intel/intel-optimized-tensorflow:latest` [↩](#a3)</sub><br/>
<sub><b id="f4">4 - </b> Downloading the datasets can take some time, you should see `XX% completed` updates for the datasets [↩](#a4)</sub>

---

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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="No" do_eval="No" print_iter=50
```

All the above commands run on a single socket of the CPU and calculate the number of cores using `BaseModelInitializer.set_num_inter_intra_threads()`.

For training in multi-instance mode (2 sockets in a single node for example) in evaluation mode,
where you are "saving checkpoints" and "doing the evaluation":

```
python launch_benchmark.py \
    --framework tensorflow \
    --precision bfloat16 \
    --mode training \
    --model-name transformer_mlperf \
    --data-location $DATA_DIR \
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
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

4.  The log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. Below are
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

# <a name="fp32-inference-instructions"></a> FP32 Inference Instructions

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
    $ export DATA_DIR=/home/<user>/transformer_data
    $ mkdir $DATA_DIR && cd $DATA_DIR
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
    ```

3. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo (from step 1).
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the dataset location (from step 2).


Before running inference, users should have the model fully trained and have saved checkpoints ready at the path $CHECKPOINT_DIR:

```
python launch_benchmark.py \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --model-name transformer_mlperf \
    --batch-size 64 \
    -i 0 --data-location $DATA_DIR \
    --checkpoint $CHECKPOINT_DIR \
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
    -- file=newstest2014.en  file_out=translate.txt reference=newstest2014.de
```
4.  The log file is saved to the value of --output-dir. if not value spacified, the log will be at the models/benchmarks/common/tensorflow/logs in workspace.
The performance and accuracy in the the log output when the benchmarking completes should look
something like this, the real throughput and inferencing time varies:
```
	Total inferencing time: xxx
	Throughput: xxx  sentences/second
	Case-insensitive results: 26.694846153259277
	Case-sensitive results: 26.182371377944946
```


## <a name="bfloat16-inference-instructions"></a> Bfloat16 Inference Instructions

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
    $ export DATA_DIR=/home/<user>/transformer_data
    $ mkdir $DATA_DIR && cd $DATA_DIR
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    $ wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
    ```
    
3. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo (from step 1).
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the dataset location (from step 2).


Before running inference, users should have the model fully trained and have saved checkpoints ready at the path $CHECKPOINT_DIR:

```
python launch_benchmark.py \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --model-name transformer_mlperf \
    --batch-size 64 \
    -i 0 --data-location $DATA_DIR \
    --checkpoint $CHECKPOINT_DIR \
    --docker-image intel/intel-optimized-tensorflow:2.4.0 \
    --verbose \
    -- file=newstest2014.en  file_out=translate.txt reference=newstest2014.de
```
The log file is saved to the value of --output-dir. if not value spacified, the log will be at the models/benchmarks/common/tensorflow/logs in workspace.
The performance and accuracy in the the log output when the benchmarking completes should look
something like this, the real throughput and inferencing time varies:
```
	Total inferencing time: xxx
	Throughput: xxx  sentences/second
	Case-insensitive results: 27.636119723320007
	Case-sensitive results: 27.127626538276672
```


