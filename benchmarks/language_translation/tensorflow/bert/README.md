# BERT

- [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions


1. Clone [intelai/models](https://github.com/IntelAI/models) repository:

   ```bash
   $ git clone https://github.com/IntelAI/models.git
   ```

   This repository includes launch scripts for running an optimized version of the BERT model code.


2. Download [facebook XNLI dataset](https://github.com/facebookresearch/XNLI) as test set and unzip it:

   ```bash
   $ mkdir data && cd data
   $ wget https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
   $ unzip XNLI-1.0.zip
   $ cd ..
   ```


3. Download the fp32 pretrained model and unzip it:

   ```bash
   $ cd XNLI-1.0
   $ wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
   $ unzip chinese_L-12_H-768_A-12.zip
   $ cd ../..
   ```


4. Download [google-research/bert](https://github.com/google-research/bert):

   ```bash
   $ git clone --single-branch https://github.com/google-research/bert.git && cd bert/
   $ git checkout 88a817c37f788702a363ff935fd173b6dc6ac0d6
   ```


5. Navigate to the `benchmarks` directory in your local clone of the [intelai/models](https://github.com/IntelAI/models) repo from step 1. Then `launch_benchmark.py` script in the `benchmarks` directory is used for starting a model. It has arguments to specify which model, framework, mode, platform, and docker image to use.
   For online inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`).

   Run with local TensorFlow:
   ```bash
   # Set dataset dir from step 2
   $ export XNLI_DIR=/home/<user>/data/XNLI-1.0
   # Set model source dir from step 4
   $ export MODEL_SOURCE=/home/<user/bert/
   
   $ cd intel-models/benchmarks
   $ python launch_benchmark.py \
     --data-location $XNLI_DIR \
     --model-source-dir $MODEL_SOURCE \
     --model-name bert \
     --precision fp32 \
     --mode inference \
     --framework tensorflow \
     --benchmark-only \
     --batch-size 1 \
     --num-cores 28 \
     --num-inter-threads 1 \
     --num-intra-threads 28 \
     --socket-id 0 \
     -- \
     task-name=XNLI \
     max-seq-length=128 \
     eval-batch-size=1 \
     learning-rate=5e-5 \
     vocab-file=$XNLI_DIR/chinese_L-12_H-768_A-12/vocab.txt \
     bert-config-file=$XNLI_DIR/chinese_L-12_H-768_A-12/bert_config.json \
     init-checkpoint=$XNLI_DIR/chinese_L-12_H-768_A-12/bert_model.ckpt
   ```

   Run with Intel optimized TensorFlow docker container:
   ```bash
   # Set dataset dir from step 2
   $ export XNLI_DIR=/home/<user>/data/XNLI-1.0
   # Set model source dir from step 4
   $ export MODEL_SOURCE=/home/<user/bert/
   # Set docker volumes for model customized parameters
   $ export DOCKER_DIR=/dataset

   $ cd intel-models/benchmarks
   $ python launch_benchmark.py \
     --data-location $XNLI_DIR \
     --model-source-dir $MODEL_SOURCE \
     --model-name bert \
     --precision fp32 \
     --mode inference \
     --framework tensorflow \
     --benchmark-only \
     --batch-size 1 \
     --num-cores 28 \
     --num-inter-threads 1 \
     --num-intra-threads 28 \
     --socket-id 0 \
     --docker-image aipg-tf/intel-optimized-tensorflow:2.0.0-mkl-py3 \
     -- \
     task-name=XNLI \
     max-seq-length=128 \
     eval-batch-size=1 \
     learning-rate=5e-5 \
     vocab-file=$DOCKER_DIR/chinese_L-12_H-768_A-12/vocab.txt \
     bert-config-file=$DOCKER_DIR/chinese_L-12_H-768_A-12/bert_config.json \
     init-checkpoint=$DOCKER_DIR/chinese_L-12_H-768_A-12/bert_model.ckpt
   ```


6. The log file is saved to the `models/benchmarks/common/tensorflow/logs` directory. Below are examples of what the tail of your log file should look like for the different configs.

   ```bash
   I1107 16:55:06.700568 140653109442368 run_classifier.py:966] ***** Eval results *****
   I1107 16:55:06.700762 140653109442368 run_classifier.py:968]   eval_accuracy = 0.6838235
   I1107 16:55:06.701230 140653109442368 run_classifier.py:968]   eval_loss = 0.6279106
   I1107 16:55:06.701292 140653109442368 run_classifier.py:968]   global_step = 408
   I1107 16:55:06.701335 140653109442368 run_classifier.py:968]   latency_per_step = 0.0682590562923
   I1107 16:55:06.701378 140653109442368 run_classifier.py:968]   latency_total = 27.8496949673
   I1107 16:55:06.701426 140653109442368 run_classifier.py:968]   loss = 0.6279106
   ```


Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands to get additional debug output or change the default output location.
