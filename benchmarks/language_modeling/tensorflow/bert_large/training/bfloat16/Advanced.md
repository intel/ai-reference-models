<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# BERT Large Bfloat16 training - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running BERT Large Bfloat16
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

export CHECKPOINT_DIR=<path to the pretrained bert model directory>
export DATASET_DIR=<path to GLUE data for classifier training or to the SQuAD data for SQuAD training>
export OUTPUT_DIR=<directory where checkpoints and log files will be saved>
```

BERT Large training can be run in three different modes:

* **To run SQuAD training** use the following command with the `train_option=SQuAD`.  The
  `CHECKPOINT_DIR` should point to the location where you've downloaded the BERT
  large (whole word masking) pretrained model, and the `DATASET_DIR` should point to
  the SQuAD data.
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \
    --volume $CHECKPOINT_DIR:$CHECKPOINT_DIR \
    --volume $DATASET_DIR:$DATASET_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --output-dir $OUTPUT_DIR \
    -- train_option=SQuAD \
       vocab_file=$CHECKPOINT_DIR/vocab.txt \
       config_file=$CHECKPOINT_DIR/bert_config.json \
       init_checkpoint=$CHECKPOINT_DIR/bert_model.ckpt \
       do_train=True \
       train_file=$DATASET_DIR/train-v1.1.json \
       do_predict=True \
       predict_file=$DATASET_DIR/dev-v1.1.json \
       learning_rate=3e-5 \
       num_train_epochs=2 \
       max_seq_length=384 \
       doc_stride=128 \
       optimized_softmax=True \
       experimental_gelu=False \
       do_lower_case=True
  ```
  The dev set predictions will be saved to a file called predictions.json in the output directory.
  ```
  python ${DATASET_DIR}/evaluate-v1.1.py ${DATASET_DIR}/dev-v1.1.json ${OUTPUT_DIR}/predictions.json
  ```
  An execution with these parameters produces results in line with below scores:
  ```
  Bf16: {"exact_match": 86.77388836329234, "f1": 92.98642358746287}
  FP32: {"exact_match": 86.72658467360453, "f1": 92.98046893150796}
  ```

  To run distributed training of SQuAD: for better throughput, simply specify `--mpi_num_processes=<num_of_sockets> [--mpi_num_processes_per_socket=1]`.
  To set `--mpi_num_processes=<num_of_sockets>`, please run `lscpu` on your machine to check the available number of sockets.
  >Note:
  >- the `global batch size` is `mpi_num_processes * train_batch_size` and sometimes `learning rate` needs to be adjusted for convergence.
  >- `square root learning rate scaling` is used by default.
  >- for BERT fine-tuning, state-of-the-art accuracy can be achieved via parallel training without synchronizing gradients between MPI workers.
  >- The `--mpi_workers_sync_gradients=[True/False]` controls whether the MPI workers sync gradients.By default it is set to `False` meaning the workers are training independently and the best performing training results will be picked in the end.
  >- To enable gradients synchronization, set the `--mpi_workers_sync_gradients` to `True` in BERT-specific options.
  >- The options `optimized_softmax=True` can be set for better performance.

  Note : Add a space after ```--```, for BERT-specific options.
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \
    --mpi_num_processes=<num_of_sockets> \
    --volume $CHECKPOINT_DIR:$CHECKPOINT_DIR \
    --volume $DATASET_DIR:$DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- train_option=SQuAD \
       vocab_file=$CHECKPOINT_DIR/vocab.txt \
       config_file=$CHECKPOINT_DIR/bert_config.json \
       init_checkpoint=$CHECKPOINT_DIR/bert_model.ckpt \
       do_train=True \
       train_file=$DATASET_DIR/train-v1.1.json \
       do_predict=True \
       predict_file=$DATASET_DIR/dev-v1.1.json \
       learning_rate=3e-5 \
       num_train_epochs=2 \
       max_seq_length=384 \
       doc_stride=128 \
       optimized_softmax=True \
       experimental_gelu=False \
       do_lower_case=True
  ```
  The results file will be written to the `${OUTPUT_DIR}` directory.

* **To run Classifier** training, use the following command with the
  `train-option=Classifier`. For classifier training, use the GLUE data in
  as the `DATASET_DIR`. The `CHECKPOINT_DIR` should point to the BERT base
  uncased 12-layer, 768-hidden pretrained model that you've downloaded.
  This example code fine-tunes BERT-Base on the Microsoft Research Paraphrase
  Corpus (MRPC) corpus, which only contains 3,600 examples.
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=32 \
    --volume $CHECKPOINT_DIR:$CHECKPOINT_DIR \
    --volume $DATASET_DIR:$DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- train-option=Classifier \
       task-name=MRPC \
       do-train=true \
       do-eval=true \
       data-dir=$DATASET_DIR/MRPC \
       vocab-file=$CHECKPOINT_DIR/vocab.txt \
       config-file=$CHECKPOINT_DIR/bert_config.json \
       init-checkpoint=$CHECKPOINT_DIR/bert_model.ckpt \
       max-seq-length=128 \
       learning-rate=2e-5 \
       num-train-epochs=30 \
       optimized_softmax=True \
       experimental_gelu=False \
       do-lower-case=True
  ```
  The results file will be written to the `${OUTPUT_DIR}` directory.

  To run distributed training of Classifier: for better throughput, specify
  `--mpi_num_processes=<num_of_sockets> [--mpi_num_processes_per_socket=1]`,
  please run `lscpu` on your machine to check the available number of sockets.
  Note that the `global batch size` is `mpi_num_processes * train_batch_size`
  and sometimes learning rate needs to be adjusted for convergence. By default,
  the script uses square root learning rate scaling.
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=32 \
    --mpi_num_processes=<num_of_sockets> \
    --volume $CHECKPOINT_DIR:$CHECKPOINT_DIR \
    --volume $DATASET_DIR:$DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- train-option=Classifier \
       task-name=MRPC \
       do-train=true \
       do-eval=true \
       data-dir=$DATASET_DIR/MRPC \
       vocab-file=$CHECKPOINT_DIR/vocab.txt \
       config-file=$CHECKPOINT_DIR/bert_config.json \
       init-checkpoint=$CHECKPOINT_DIR/bert_model.ckpt \
       max-seq-length=128 \
       learning-rate=2e-5 \
       num-train-epochs=30 \
       optimized_softmax=True \
       experimental_gelu=True \
       do-lower-case=True
  ```
  The results file will be written to the `${OUTPUT_DIR}` directory.
* **To run pre-training from scratch**, use the following command with
  `train-option=Pretraining`. Pre-training has two phases:
  In the first phase, the data is generated for sequential length 128.
  And in the second phase, sequential length 512 is used.
  Please follow instructions in [google bert pre-training](https://github.com/google-research/bert#pre-training-with-bert)
  for data pre-processing and set the `DATASET_DIR` environment variable
  to the TF record directory. The `CHECKPOINT_DIR` should point to the
  BERT large uncased (whole word masking) directory.
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \
    --socket-id=0 \
    --num-intra-threads=24 \
    --num-inter-threads=1 \
    --volume $CHECKPOINT_DIR:$CHECKPOINT_DIR \
    --volume $DATASET_DIR:$DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- train-option=Pretraining \
       input-file=$DATASET_DIR/tf_wiki_formatted_slen512.tfrecord \
       do-train=True \
       do-eval=True \
       config-file=$CHECKPOINT_DIR/bert_config.json \
       init-checkpoint=$CHECKPOINT_DIR/bert_model.ckpt \
       max-seq-length=512 \
       max-predictions=76 \
       num-train-steps=20 \
       warmup-steps=10 \
       learning-rate=2e-5 \
       optimized_softmax=True \
       experimental_gelu=False \
       profile=False
  ```

  To run distributed training of pretraining: for better throughput, simply specify
  `--mpi_num_processes=<num_of_sockets> [--mpi_num_processes_per_socket=1]`. Please
  run `lscpu` on your machine to check the available number of sockets.
  > Note that the `global batch size` is `mpi_num_processes * train_batch_size` and
  > sometimes `learning rate` needs to be adjusted for convergence. By default, the
  > script uses `square root learning rate scaling`.
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \
    --num-intra-threads=22 \
    --num-inter-threads=1 \
    --mpi_num_processes=4 \
    --volume $CHECKPOINT_DIR:$CHECKPOINT_DIR \
    --volume $DATASET_DIR:$DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- train-option=Pretraining \
       input-file=$DATASET_DIR/tf_wiki_formatted_slen512.tfrecord \
       do-train=True \
       do-eval=True \
       config-file=$CHECKPOINT_DIR/bert_config.json \
       init-checkpoint=$CHECKPOINT_DIR/bert_model.ckpt \
       max-seq-length=512 \
       max-predictions=76 \
       num-train-steps=20 \
       warmup-steps=10 \
       learning-rate=2e-5 \
       optimized_softmax=True \
       experimental_gelu=False \
       profile=False
  ```

  >Note: for best performance, we will set `num-intra-thread` as follows:
  >- For single instance (mpi_num_processes=1) run: the value is equal to
  > number of logical cores per socket.
  >- For multi-instance (mpi_num_processes>1) run: the value is equal to
  > (num_of_logical_cores_per_socket - 2)

