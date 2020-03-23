# BERT

This document has instructions for how to run BERT for the
following modes/platforms:
* [BFloat16 training](#training-instructions)

This repo contains model for BERT Large optmized for bfloat16 and fp32 on Intel CPUs. For all fine-tunining the datasets (SQuAD, MultiNLI, and MRPC) should be downloaded as mentioned in the Google bert repo.

## BERT Environment Variables
Running BERT may need the right env variables for different tasks : SQuAD and Classifier and Pretraining. 
An example setting is below.
```
export BERT_LARGE_DIR=/path/to/bert/wwm_cased_L-24_H-1024_A-16
 # For Classifiers
export GLUE_DIR=/path/to/glue 
export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier
```
Refer to google reference page for approriate settings needed for each type of task.

## Training Instructions

 1. Download the intel-model zoo for bfloat16
       ```git clone https://github.com/NervanaSystems/intel-models```
 2. Download data for BERT from [google bert repo]((https://github.com/google-research/bert). Keep all data under one directory.
 
 3. **To run SQUAD** 
 Change dir to bechmarks and run the following command.
 
  Note : Add space after ```--```, for BERT options.

```
# These simplify the settings
export BERT_LARGE_DIR=/path/to/bert/wwm_cased_L-24_H-1024_A-16
export SQUAD_DIR=/path/to/bert/SQuAD

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \ 
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
       output_dir=./large \
       do_lower_case=False

```
Please refer to google docs for SQuAD specific arguments.

4. **To run Classifier**
         ```train-option=Classifier```
```
export BERT_LARGE_DIR=/path/to/bert/wwm_cased_L-24_H-1024_A-16
export GLUE_DIR=/path/to/glue 

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=32 \
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
       output-dir=/tmp/mrpc_output/ \
       do-lower-case=False

```
 
 **5. To run pre-training** 
 ***(This is still experimental).*** Need to run pretraining of data as discussed in original google bert location).
        Replace SQuAD with "Pretraining" in --train_option and use the right options for Pretraining from Google BERT"
           ```train-option=Pretraining```
As shown below
```
export BERT_LARGE_DIR=/path/to/bert/wwm_cased_L-24_H-1024_A-16

python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=32 \
    -- train-option=Pretraining \
       input-file=/tmp/tf_examples.tfrecord \
       output-dir=/tmp/pretraining_output \
       do-train=True \
       do-eval=True \
       config-file=$BERT_LARGE_DIR/bert_config.json \
       init-checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
       max-seq-length=128 \
       max-predictions=20 \
       num-train-steps=20 \
       warmup-steps=10 \
       learning-rate=2e-5
  

```
