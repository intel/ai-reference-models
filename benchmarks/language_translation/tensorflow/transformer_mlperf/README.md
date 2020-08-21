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

<sub><b id="f1">1 - </b> Running `python data_download.py --data_dir=$DATA_DIR` assumes you have a python environment similar to what the intel/intel-optimized-tensorflow:2.3.0 container provides. One option would be to run the above within the intel/intel-optimized-tensorflow:2.3.0 container eg: `docker run -u $(id -u):$(id -g) --privileged  --entrypoint /bin/bash -v /home/<user>:/home/<user> -it intel/intel-optimized-tensorflow:latest` [↩](#a1)</sub><br/>
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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

## Bfloat16 Training Instructions

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

<sub><b id="f3">3 - </b> Running `python data_download.py --data_dir=$DATA_DIR` assumes you have a python environment similar to what the intel/intel-optimized-tensorflow:2.3.0 container provides. One option would be to run the above within the intel/intel-optimized-tensorflow:2.3.0 container eg: `docker run -u $(id -u):$(id -g) --privileged  --entrypoint /bin/bash -v /home/<user>:/home/<user> -it intel/intel-optimized-tensorflow:latest` [↩](#a3)</sub><br/>
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
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

Example log tail when running training without saving checkpoints and not doing the evaluations:
```
I0820 22:23:04.168998 139758652872512 basic_session_run_hooks.py:260] loss = 8.073875 (631.692 sec)
INFO:tensorflow:Loss for final step: 7.7058005.
I0820 22:33:17.025216 139758652872512 estimator.py:350] Loss for final step: 7.7058005.
:::MLPv0.5.0 transformer 1597962797.026246309 (transformer/transformer_main.py:353) eval_start
I0820 22:33:17.026297 139758652872512 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597962797.026246309 (transformer/transformer_main.py:353) eval_start
:::MLPv0.5.0 transformer 1597962797.026616573 (transformer/transformer_main.py:369) eval_stop
I0820 22:33:17.026626 139758652872512 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597962797.026616573 (transformer/transformer_main.py:369) eval_stop
:::MLPv0.5.0 transformer 1597962797.026885271 (transformer/transformer_main.py:453) run_stop
I0820 22:33:17.026893 139758652872512 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597962797.026885271 (transformer/transformer_main.py:453) run_stop
:::MLPv0.5.0 transformer 1597962797.027143478 (transformer/transformer_main.py:454) run_final
I0820 22:33:17.027151 139758652872512 mlperf_log.py:134] :::MLPv0.5.0 transformer 1597962797.027143478 (transformer/transformer_main.py:454) run_final
num_inter_threads: 1
num_intra_threads: 28
Received these standard args: Namespace(accuracy_only=False, backbone_model=None, batch_size=-1, benchmark_dir='/workspace/benchmarks', benchmark_only=True, bleu_ref=None, bleu_source=None, bleu_threshold=None, checkpoint=None, data_location='/dataset', data_num_inter_threads=None, data_num_intra_threads=None, disable_tcmalloc=True, do_eval='No', epochs_between_eval=1, experimental_gelu=False, framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', learning_rate=2, mode='training', model_args=[], model_name='transformer_mlperf', model_source_dir='/workspace/models', mpi=None, mpi_hostnames=None, num_cores=-1, num_cpu_cores=4, num_instances=1, num_inter_threads=1, num_intra_threads=28, num_mpi=1, num_train_steps=1, optimized_softmax=True, output_dir='/workspace/benchmarks/common/tensorflow/logs', output_results=False, params='big', precision='bfloat16', print_iter=50, random_seed=11, save_checkpoints='No', save_profile='', socket_id=0, static_batch='No', steps_between_eval=200, tcmalloc_large_alloc_report_threshold=2147483648, tf_serving_version='master', train_epochs=None, train_steps=200, use_case='language_translation', verbose=True)
Received these custom args: ['--random_seed=11', '--params=big', '--train_steps=200', '--steps_between_eval=200', '--do_eval=No', '--save_checkpoints=No', '--print_iter=50', '--save_profile=']
Current directory: /workspace/benchmarks
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/training/bfloat16/transformer/transformer_main.py --data_dir=/dataset --random_seed=11 --params=big --train_steps=200 --steps_between_eval=200 --do_eval=No --save_checkpoints=No --save_profile= --print_iter=50 --inter_op_parallelism_threads=1 --intra_op_parallelism_threads=28 --learning_rate=2 --static_batch=No
PYTHONPATH: :/workspace/intelai_models_common:/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=language_translation --model-name=transformer_mlperf --precision=bfloat16 --mode=training --benchmark-dir=/workspace/benchmarks --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=-1 --socket-id=0 --output-dir=/workspace/benchmarks/common/tensorflow/logs --num-train-steps=1  --benchmark-only  --verbose --model-source-dir=/workspace/models --data-location=/dataset --disable-tcmalloc=True --random_seed=11 --params=big --train_steps=200 --steps_between_eval=200 --do_eval=No --save_checkpoints=No
    --print_iter=50 --save_profile=
Log file location: /tmp/models/benchmarks/common/tensorflow/logs/benchmark_transformer_mlperf_training_bfloat16_20200820_214732.log
```

Multi-instance sample log:
Running 2 instances only. Both instances printing loss values.

```
INFO:tensorflow:loss = 10.301315, step = 0
I0820 21:13:46.964926 140325266102080 basic_session_run_hooks.py:262] loss = 10.301315, step = 0
INFO:tensorflow:loss = 10.301315
I0820 21:13:46.965517 140325266102080 basic_session_run_hooks.py:262] loss = 10.301315
INFO:tensorflow:loss = 10.317959, step = 0
I0820 21:13:46.974031 140221457520448 basic_session_run_hooks.py:262] loss = 10.317959, step = 0
INFO:tensorflow:loss = 10.317959
I0820 21:13:46.974589 140221457520448 basic_session_run_hooks.py:262] loss = 10.317959
INFO:tensorflow:Batch [50]:  last 50 steps exp/sec = 372.306, completed 50/50 wamrup steps
I0820 21:25:14.571191 140325266102080 hooks.py:117] Batch [50]:  last 50 steps exp/sec = 372.306, completed 50/50 wamrup steps
INFO:tensorflow:loss = 9.06568 (687.606 sec)
I0820 21:25:14.571795 140325266102080 basic_session_run_hooks.py:260] loss = 9.06568 (687.606 sec)
INFO:tensorflow:Batch [50]:  last 50 steps exp/sec = 372.309, completed 50/50 wamrup steps
I0820 21:25:14.574600 140221457520448 hooks.py:117] Batch [50]:  last 50 steps exp/sec = 372.309, completed 50/50 wamrup steps
INFO:tensorflow:loss = 8.979176 (687.601 sec)
I0820 21:25:14.575186 140221457520448 basic_session_run_hooks.py:260] loss = 8.979176 (687.601 sec)
:::MLPv0.5.0 [08c275d5-fd28-41be-bb3c-948dd6a7c8d3][1597959384.6937411][3.08816198e-06]
:::MLPv0.5.0 [259380ab-382d-40e9-8971-1e0fd1fd5006][1597959384.701951][3.08816198e-06]
INFO:tensorflow:global_step/sec: 0.0728256
I0820 21:36:40.117180 140221457520448 basic_session_run_hooks.py:702] global_step/sec: 0.0728256
INFO:tensorflow:Batch [100]:  last 50 steps exp/sec = 373.426, total average exp/sec = 373.426
I0820 21:36:40.118289 140221457520448 hooks.py:113] Batch [100]:  last 50 steps exp/sec = 373.426, total average exp/sec = 373.426
INFO:tensorflow:loss = 8.15804, step = 100 (1373.145 sec)
I0820 21:36:40.118491 140221457520448 basic_session_run_hooks.py:260] loss = 8.15804, step = 100 (1373.145 sec)
INFO:tensorflow:loss = 8.15804 (685.543 sec)
I0820 21:36:40.118592 140221457520448 basic_session_run_hooks.py:260] loss = 8.15804 (685.543 sec)
INFO:tensorflow:global_step/sec: 0.072825
I0820 21:36:40.120503 140325266102080 basic_session_run_hooks.py:702] global_step/sec: 0.072825
INFO:tensorflow:Batch [100]:  last 50 steps exp/sec = 373.423, total average exp/sec = 373.423
I0820 21:36:40.121415 140325266102080 hooks.py:113] Batch [100]:  last 50 steps exp/sec = 373.423, total average exp/sec = 373.423
INFO:tensorflow:loss = 8.131919, step = 100 (1373.157 sec)
I0820 21:36:40.121570 140325266102080 basic_session_run_hooks.py:260] loss = 8.131919, step = 100 (1373.157 sec)
INFO:tensorflow:loss = 8.131919 (685.550 sec)
I0820 21:36:40.121664 140325266102080 basic_session_run_hooks.py:260] loss = 8.131919 (685.550 sec)
```
