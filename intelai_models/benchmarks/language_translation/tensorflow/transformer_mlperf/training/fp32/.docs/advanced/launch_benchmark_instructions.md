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

Note users can specify batch size suitable to their systems with command option `--batch-size` to achieve the best performance. If users don't specify the batch size, then the model will choose default batch size for training, which is set to 5120 in the model.
For training with all epochs, with evaluation, and with saving checkpoints (for convergence):
```
python launch_benchmark.py \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --model-name transformer_mlperf \
    --socket-id 0 \
    --data-location ${DATASET_DIR} \
    --docker-image <docker image>  \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    -- random_seed=11 train_steps=0 steps_between_eval=0 params=big save_checkpoints="Yes" do_eval="Yes" print_iter=50
```

For training with fewer training steps (200 steps), and with no evaluation:
```
python launch_benchmark.py \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --model-name transformer_mlperf \
    --socket-id 0 \
    --data-location ${DATASET_DIR} \
    --docker-image <docker image>  \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="Yes" do_eval="No" print_iter=50
```

For training with fewer training steps (200 steps), and with evaluation:
```
python launch_benchmark.py \
    --framework tensorflow \
    --precision fp32 \
    --mode training \
    --model-name transformer_mlperf \
    --socket-id 0 \
    --data-location ${DATASET_DIR} \
    --docker-image <docker image>  \
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
    --precision fp32 \
    --mode training \
    --model-name transformer_mlperf \
    --socket-id 0 \
    --data-location ${DATASET_DIR} \
    --docker-image <docker image>  \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
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
    --data-location ${DATASET_DIR} \
    --docker-image <docker image> \
    --batch-size ${BATCH_SIZE} \
    --num-intra-threads=26 --num-inter-threads=1 \
    --mpi_num_processes=2 \
    --output-dir ${OUTPUT_DIR} \
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
    --data-location ${DATASET_DIR} \
    --docker-image <docker image>  \
    --batch-size ${BATCH_SIZE} \
    --num-intra-threads=26 --num-inter-threads=1 \
    --mpi_num_processes=2 \
    --output-dir ${OUTPUT_DIR} \
    -- random_seed=11 train_steps=200 steps_between_eval=200 params=big save_checkpoints="No" do_eval="No" print_iter=50
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output. The above two command assumes 2 sockets in the CPU,
and 28 cores per socket. We keep aside 2 cores for horovod communications. Thus,
in general the num-intra-threads is equal to "available cores per socket - 2".
The transformer model related parameters is appended after "-- "

The log file is saved to the `$OUTPUT_DIR` directory. Below are
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
