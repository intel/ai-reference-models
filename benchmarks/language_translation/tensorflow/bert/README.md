# BERT

- [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions

Clone [intelai/models](https://github.com/IntelAI/models) repository:

   ```
   git clone https://github.com/IntelAI/models.git
   ```
This repository includes launch scripts for running an optimized version of the BERT model code.


### Dataset
Download Microsoft Research Paraphrase Corpus (MRPC) data in cloned repository and save it inside `data` folder.
You can also use the helper script [download_glue_data.py](https://gist.github.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3) to download the data:

   ```
   # Obtain a copy of download_glue_data.py to the current directory
   wget https://gist.githubusercontent.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3/raw/db67cdf22eb5bd7efe376205e8a95028942e263d/download_glue_data.py
   python3 download_glue_data.py --data_dir ./data/ --tasks MRPC
   ```


### Run the model
* Download the fp32 "BERT-Base, Uncased" pre-trained model and unzip it inside the dataset directory `data/MRPC`.
If you run on Windows, please use a browser to download the pretrained model using the link below. For Linux, run:

   ```
   cd data/MRPC
   wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
   unzip uncased_L-12_H-768_A-12.zip
   cd ../..
   ```

* Clone [google-research/bert](https://github.com/google-research/bert) repository:
   ```
   git clone --single-branch https://github.com/google-research/bert.git && cd bert/
   git checkout 88a817c37f788702a363ff935fd173b6dc6ac0d6
   ```
   Set the `MODEL_SOURCE` environment variable to the location of `bert` directory.
   
* Install [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/) on your system.

* Navigate to the `benchmarks` directory in your local clone of the [intelai/models](https://github.com/IntelAI/models) repo.
Then `launch_benchmark.py` script in the `benchmarks` directory is used for starting a model.
It has arguments to specify which model, framework, mode, precision, and docker image to use.
Small sets like MRPC have a high variance in the Dev set accuracy, even when starting from the same pre-training checkpoint.
If you re-run multiple times (making sure to point to different `output-dir`), you should see results between 84% and 88%.
By default model will run in performance mode without fine-tuning to save time.
Using `--accuracy-only` to enable complete process and get right accuracy.
For online inference (using `--socket-id 0` and `--batch-size 1`).
For throughput, recommend batch size is 32 (using `--batch-size 32`).

   #### Run on Linux:
   ```
   # Set dataset dir
   export MRPC_DIR=/home/<user>/data/MRPC
   # Set model source dir
   export MODEL_SOURCE=/home/<user>/bert/
   
   cd models/benchmarks
   python launch_benchmark.py \
     --accuracy-only \
     --checkpoint $MRPC_DIR/uncased_L-12_H-768_A-12/ \
     --data-location $MRPC_DIR \
     --model-source-dir $MODEL_SOURCE \
     --model-name bert \
     --precision fp32 \
     --mode inference \
     --framework tensorflow \
     --batch-size 1 \
     --num-cores 28 \
     --num-inter-threads 1 \
     --num-intra-threads 28 \
     --socket-id 0 \
     --output-dir $MRPC_DIR/tmp \
     -- \
     task-name=MRPC \
     max-seq-length=128 \
     learning-rate=2e-5 \
     num_train_epochs=3.0
   ```
   Use `--docker-image intel/intel-optimized-tensorflow:latest` to run with Intel optimized TensorFlow docker container.
   
   #### Run on Windows:
   If not already setup, please follow instructions for [environment setup on Windows](/docs/general/tensorflow/Windows.md).
   Then, run inference using `cmd.exe`:
   ```
   # Set dataset dir
   set MRPC_DIR=C:\\<user>\\data\\MRPC
   # Set model source dir
   set MODEL_SOURCE=C:\\<user>\\bert
   python launch_benchmark.py ^
      --accuracy-only ^
      --checkpoint  %MRPC_DIR%\\uncased_L-12_H-768_A-12 ^
      --data-location %MRPC_DIR% ^
      --model-source-dir %MODEL_SOURCE% ^
      --model-name bert ^
      --precision fp32 ^
      --mode inference ^
      --framework tensorflow ^
      --batch-size 1 ^
      --num-cores 28 ^
      --num-inter-threads 1 ^
      --num-intra-threads 28 ^
      --output-dir  C:\\<user>\\MRPC\\tmp ^
      --verbose ^
      -- ^
      task-name=MRPC ^
      max-seq-length=128 ^
      learning-rate=2e-5 ^
      num_train_epochs=3.0
   ```

* The log file is saved to the `models/benchmarks/common/tensorflow/logs` directory. Below are examples of what the tail of your log file should look like for the different configs.

   ```
   I0120 14:15:19.862291 140620068984640 run_classifier.py:968] ***** Eval results *****
   I0120 14:15:19.862353 140620068984640 run_classifier.py:970]   eval_accuracy = 0.85294116
   I0120 14:15:19.862770 140620068984640 run_classifier.py:970]   eval_loss = 0.51752657
   I0120 14:15:19.862843 140620068984640 run_classifier.py:970]   global_step = 408
   I0120 14:15:19.862894 140620068984640 run_classifier.py:970]   latency_per_step = 0.0811467521331
   I0120 14:15:19.862942 140620068984640 run_classifier.py:970]   latency_total = 33.1078748703
   I0120 14:15:19.862983 140620068984640 run_classifier.py:970]   loss = 0.51752657
   ```

   Thoughput will be displayed below if batch size is greater than 1:
   ```
   I0120 14:38:12.126959 140150726125376 run_classifier.py:967] ***** Eval results *****
   I0120 14:38:12.127017 140150726125376 run_classifier.py:969]   eval_accuracy = 0.8489583
   I0120 14:38:12.127545 140150726125376 run_classifier.py:969]   eval_loss = 0.534228
   I0120 14:38:12.127612 140150726125376 run_classifier.py:969]   global_step = 3
   I0120 14:38:12.127656 140150726125376 run_classifier.py:969]   latency_per_step = 4.69783433278
   I0120 14:38:12.127700 140150726125376 run_classifier.py:969]   latency_total = 14.0935029984
   I0120 14:38:12.127742 140150726125376 run_classifier.py:969]   loss = 0.534228
   I0120 14:38:12.127783 140150726125376 run_classifier.py:969]   samples_per_sec = 27.2465972473
   ```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands to get additional debug output or change the default output location.
