# BERT

This document has instructions for how to run [BERT](https://github.com/google-research/bert#what-is-bert) for the
following modes/platforms:
* [BFloat16 training](#bfloat16-training-instructions)
* [FP32 training](#fp32-training-instructions)
* [BFloat16 inference](#bfloat16-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)


For all fine-tuning, the datasets (SQuAD, MultiNLI, MRPC etc..) and checkpoints should be downloaded as mentioned in the [Google bert repo](https://github.com/google-research/bert).

Refer to google reference page for [checkpoints](https://github.com/google-research/bert#pre-trained-models).

## BERT Environment Variables
When running BERT for different tasks : SQuAD, Classifiers and Pretraining setting env variables can simplify the commands. 

An example setting is below.
```
export BERT_LARGE_DIR=/home/<user>/wwm_uncased_L-24_H-1024_A-16
# For Classifiers
export GLUE_DIR=/home/<user>/glue 
#For SQuAD
export SQUAD_DIR=/home/<user>/SQUAD
```

## BFloat16 Training Instructions
(Experimental)

 1. Clone the intelai/models repo.
This repo has the launch script for running the model, which we will
use in the next step.
```
git clone https://github.com/IntelAI/models.git
```
 2. Download checkpoints and data for BERT from [google bert repo](https://github.com/google-research/bert).
    Keep all data under one directory ([SQuAD](https://github.com/google-research/bert#squad-11), GLUE). For training from scratch Wikipedia and BookCorpus need to be downloaded and pre-processed.
 3. **To run SQuAD** 
    Navigate to `models/benchmarks` directory and run the following command:

Note : Add space after `--`, for BERT-specific options.

```
export BERT_LARGE_DIR=/home/<user>/wwm_uncased_L-24_H-1024_A-16
export SQUAD_DIR=/home/<user>/SQuAD

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --volume $BERT_LARGE_DIR:$BERT_LARGE_DIR \
    --volume $SQUAD_DIR:$SQUAD_DIR \
    -- train_option=SQuAD \
       vocab_file=$BERT_LARGE_DIR/vocab.txt \
       config_file=$BERT_LARGE_DIR/bert_config.json \
       init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
       do_train=True \
       train_file=$SQUAD_DIR/train-v1.1.json \
       do_predict=True \
       predict_file=$SQUAD_DIR/dev-v1.1.json \
       learning_rate=3e-5 \
       num_train_epochs=2 \
       max_seq_length=384 \
       doc_stride=128 \
       optimized_softmax=True \
       experimental_gelu=False \
       do_lower_case=True

```
The results file will be written to the
`models/benchmarks/common/tensorflow/logs` directory, unless another
output directory is specified by the `--output-dir` arg.

The dev set predictions will be saved to a file called `predictions.json` in the output directory.

```
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /home/<user>/models/benchmarks/common/tensorflow/logs/predictions.json
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

Navigate to `models/benchmarks` directory and run the following command:
 
  Note : Add space after ```--```, for BERT-specific options.

```
export BERT_LARGE_DIR=/home/<user>/wwm_uncased_L-24_H-1024_A-16
export SQUAD_DIR=/home/<user>/SQuAD

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \ 
    --mpi_num_processes=<num_of_sockets> \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --volume $BERT_LARGE_DIR:$BERT_LARGE_DIR \
    --volume $SQUAD_DIR:$SQUAD_DIR \
    -- train_option=SQuAD \
       vocab_file=$BERT_LARGE_DIR/vocab.txt \
       config_file=$BERT_LARGE_DIR/bert_config.json \
       init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
       do_train=True \
       train_file=$SQUAD_DIR/train-v1.1.json \
       do_predict=True \
       predict_file=$SQUAD_DIR/dev-v1.1.json \
       learning_rate=3e-5 \
       num_train_epochs=2 \
       max_seq_length=384 \
       doc_stride=128 \
       optimized_softmax=True \
       experimental_gelu=False \
       do_lower_case=True
```
The results file will be written to the
`models/benchmarks/common/tensorflow/logs` directory, unless another
output directory is specified by the `--output-dir` arg.

4. **To run Classifier**
         ```train-option=Classifier```
      Download [GLUE](https://gluebenchmark.com/tasks) data by running the [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
      This example code fine-tunes BERT-Base on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples.
```
export BERT_BASE_DIR=/home/<user>/wwm_uncased_L-12_H-768_A-16
export GLUE_DIR=/home/<user>/glue 

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=32 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --volume $BERT_BASE_DIR:$BERT_BASE_DIR \
    --volume $GLUE_DIR:$GLUE_DIR \
    -- train-option=Classifier \
       task-name=MRPC \
       do-train=true \
       do-eval=true \
       data-dir=$GLUE_DIR/MRPC \
       vocab-file=$BERT_BASE_DIR/vocab.txt \
       config-file=$BERT_BASE_DIR/bert_config.json \
       init-checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
       max-seq-length=128 \
       learning-rate=2e-5 \
       num-train-epochs=30 \
       optimized_softmax=True \
       experimental_gelu=False \
       do-lower-case=True

```
The results file will be written to the
`models/benchmarks/common/tensorflow/logs` directory, unless another
output directory is specified by the `--output-dir` arg.


To run distributed training of Classifier: for better throughput, specify `--mpi_num_processes=<num_of_sockets> [--mpi_num_processes_per_socket=1]`, please run `lscpu` on your machine to check the available number of sockets.
Note that the `global batch size` is `mpi_num_processes * train_batch_size` and sometimes learning rate needs to be adjusted for convergence. By default, the script uses square root learning rate scaling.
```
export BERT_LARGE_DIR=/home/<user>/wwm_uncased_L-24_H-1024_A-16
export GLUE_DIR=/home/<user>/glue 

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=32 \
    --mpi_num_processes=<num_of_sockets> \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --volume $BERT_LARGE_DIR:$BERT_LARGE_DIR \
    --volume $GLUE_DIR:$GLUE_DIR \
    -- train-option=Classifier \
       task-name=MRPC \
       do-train=true \
       do-eval=true \
       data-dir=$GLUE_DIR/MRPC \
       vocab-file=$BERT_LARGE_DIR/vocab.txt \
       config-file=$BERT_LARGE_DIR/bert_config.json \
       init-checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
       max-seq-length=128 \
       learning-rate=2e-5 \
       num-train-epochs=30 \
       optimized_softmax=True \
       experimental_gelu=True \
       do-lower-case=True

```
 The results file will be written to the
`models/benchmarks/common/tensorflow/logs` directory, unless another
output directory is specified by the `--output-dir` arg.


 **5. To run** 
 ***Pre-training from scratch.*** Pre-training has two phases:
       In the first phase, the data is generated for sequential length 128. And in the second phase, sequential length 512 is used.
       Please follow instructions in [google bert pre-training](https://github.com/google-research/bert#pre-training-with-bert) for data pre-processing.
       Replace SQuAD with `Pretraining` in `--train_option` and set ```train-option=Pretraining```.
```
export BERT_LARGE_DIR=/home/<user>/wwm_uncased_L-24_H-1024_A-16
export PRETRAINING_DATA_DIR=/home/<user>/pretraining/tf-record-diretory

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \
    --socket-id=0 \
    --num-intra-threads=24 \
    --num-inter-threads=1 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --volume $BERT_LARGE_DIR:$BERT_LARGE_DIR \
    --volume $PRETRAINING_DATA_DIR:$PRETRAINING_DATA_DIR \
    -- train-option=Pretraining \
       input-file=$PRETRAINING_DATA_DIR/tf_wiki_formatted_slen512.tfrecord \
       output-dir=/tmp/pretraining_output \
       do-train=True \
       do-eval=True \
       config-file=$BERT_LARGE_DIR/bert_config.json \
       init-checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
       max-seq-length=512 \
       max-predictions=76 \
       num-train-steps=20 \
       warmup-steps=10 \
       learning-rate=2e-5 \
       optimized_softmax=True \
       experimental_gelu=False \
       profile=False 
```

To run distributed training of pretraining: for better throughput, simply specify `--mpi_num_processes=<num_of_sockets> [--mpi_num_processes_per_socket=1]`. Please run `lscpu` on your machine to check the available number of sockets.
>Note that the `global batch size` is `mpi_num_processes * train_batch_size` and sometimes `learning rate` needs to be adjusted for convergence. By default, the script uses `square root learning rate scaling`.
```
export BERT_LARGE_DIR=/home/<user>/wwm_uncased_L-24_H-1024_A-16
export PRETRAINING_DATA_DIR=/home/<user>/pretraining/tf-record-diretory

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \
    --num-intra-threads=22 \
    --num-inter-threads=1 \
    --mpi_num_processes=4 \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --volume $BERT_LARGE_DIR:$BERT_LARGE_DIR \
    --volume $PRETRAINING_DATA_DIR:$PRETRAINING_DATA_DIR \
    -- train-option=Pretraining \
       input-file=$PRETRAINING_DATA_DIR/tf_wiki_formatted_slen512.tfrecord \
       output-dir=/tmp/pretraining_output \
       do-train=True \
       do-eval=True \
       config-file=$BERT_LARGE_DIR/bert_config.json \
       init-checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
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
>- For single instance (mpi_num_processes=1) run: the value is equal to number of logical cores per socket.
>- For multi-instance (mpi_num_processes>1) run: the value is equal to (num_of_logical_cores_per_socket - 2)

## FP32 Training Instructions
FP32 training instructions are the same as Bfloat16 training instructions above, except one needs to change the `--precision=bfloat16` to `--precision=fp32` in the above commands.

## BFloat16 Inference Instructions
(Experimental)

 1. Clone the Intel model zoo:
    ```
    git clone https://github.com/IntelAI/models.git
    ```

 2. Download and unzip the BERT large uncased (whole word masking) model from the
    [google bert repo](https://github.com/google-research/bert#pre-trained-models).
    Then, download the `dev-v1.1.json` file from the
    [google bert repo](https://github.com/google-research/bert#squad-11)
    into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

    ```
    wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
    unzip wwm_uncased_L-24_H-1024_A-16.zip

    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
    ```

    The `wwm_uncased_L-24_H-1024_A-16` directory is what will be passed as
    the `--data-location` when running inference in step 4.

 3. Download and unzip the pretrained model:
  
    ```
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
    unzip bert_large_checkpoints.zip
    ```

    This directory will be passed as the `--checkpoint` location when
    running inference in step 4.

 4. Navigate to the `benchmarks` directory in the model zoo that you
    cloned in step 1. This directory contains the
    [launch_benchmarks.py script](/models/docs/general/tensorflow/LaunchBenchmark.md)
    that will be used to run inference.
    ```
    cd benchmarks
    ```

    Bert large inference can be run in 3 different modes:
    * **Benchmark**
        ```
        python launch_benchmark.py \
            --model-name=bert_large \
            --precision=bfloat16 \
            --mode=inference \
            --framework=tensorflow \
            --batch-size=32 \
            --data-location /home/<user>/wwm_uncased_L-24_H-1024_A-16 \
            --checkpoint /home/<user>/bert_large_checkpoints \
            --output-dir /home/<user>/bert-squad-output \
            --benchmark-only \
            --docker-image intel/intel-optimized-tensorflow:2.3.0 \
            -- infer_option=SQuAD
        ```

    * **Profile**
        ```
        python launch_benchmark.py \
            --model-name=bert_large \
            --precision=bfloat16 \
            --mode=inference \
            --framework=tensorflow \
            --batch-size=32 \
            --data-location /home/<user>/wwm_uncased_L-24_H-1024_A-16 \
            --checkpoint /home/<user>/bert_large_checkpoints \
            --output-dir /home/<user>/bert-squad-output \
            --docker-image intel/intel-optimized-tensorflow:2.3.0 \
            -- profile=True infer_option=SQuAD
        ```

    * **Accuracy**
        ```
        python launch_benchmark.py \
            --model-name=bert_large \
            --precision=bfloat16 \
            --mode=inference \
            --framework=tensorflow \
            --batch-size=32 \
            --data-location /home/<user>/wwm_uncased_L-24_H-1024_A-16 \
            --checkpoint /home/<user>/bert_large_checkpoints \
            --output-dir /home/<user>/bert-squad-output \
            --docker-image intel/intel-optimized-tensorflow:2.3.0 \
            --accuracy-only \
            -- infer_option=SQuAD
        ```

    Output files and logs are saved to the
    `models/benchmarks/common/tensorflow/logs` directory. To change the
    location, use the `--output-dir` flag in the
    [launch_benchmarks.py script](/models/docs/general/tensorflow/LaunchBenchmark.md).

    Note that args specific to this model are specified after ` -- ` at
    the end of the command (like the `profile=True` arg in the Profile
    command above. Below is a list of all of the model specific args and
    their default values:

    | Model arg | Default value |
    |-----------|---------------|
    | doc_stride | `128` |
    | max_seq_length | `384` |
    | profile | `False` |
    | config_file | `bert_config.json` |
    | vocab_file | `vocab.txt` |
    | predict_file | `dev-v1.1.json` |
    | init_checkpoint | `model.ckpt-3649` |

## FP32 Inference Instructions
FP32 inference instructions are the same as Bfloat16 inference instructions above, except one needs to change the `--precision=bfloat16` to `--precision=fp32` in the above commands.
