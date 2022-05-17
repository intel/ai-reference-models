<!--- 0. Title -->
# Transformer Language INT8 Inference

<!-- 10. Description -->
## Description

This document has instructions for running Transformer Language int8 Inference in mlperf
Benchmark suits using Intel-optimized TensorFlow.

Detailed information on mlperf Benchmark can be found in [mlcommons/training](https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)

The inference code is based on the trasnformer mlperf evaluation code, but Intel has optimized the inference model by modify the code of the model, so that it can achieve better performance on Intel CPUs.
The qunatized model is generated with Intel [LPOT tool](https://github.com/intel/neural-compressor) from fp32 model, and Intel also has optimized the model in the process of quantization. 

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATA_DIR` to point out to the location of the dataset directory.

## Run the model on Linux

Before running inference, users should have the quantized model pb files downloaded from the trained models website the Intel published.

The inference can be run with accuracy mode or benchmark mode. The benchmark mode will run with the best performance by setting warmup steps and the total steps users want to run. The accuracy mode will just run for testing accuracy without setting warmup steps and steps.

Set the environment variables to point to the dataset directory `DATA_DIR`, the pretrained model path `PB_FILE`, batch size `BATCH_SIZE`, the number of sockets `NUM_SOCKETS`, and the number of cores on your system `NUM_CORES`.
```
export PB_FILE=<path to the frozen pre trained model file>
export DATA_DIR=<the input data directory, which should include newstest2014.en, newstest2014.de and vocab.ende.32768>
export BATCH_SIZE=1
export NUM_SOCKETS=2
```
#### Benchmark mode:
```
  python3 ./benchmarks/launch_benchmark.py    \
     --benchmark-only --framework tensorflow  \
     --in-graph=$PB_FILE \
     --model-name transformer_mlperf \
     --mode inference --precision int8 \
     --batch-size $BATCH_SIZE \
     --num-intra-threads $NUM_CORES --num-inter-threads $NUM_SOCKETS \
     --verbose \
     --data-location $DATA_DIR \
     --docker-image intel/intel-optimized-tensorflow:latest \
     -- params=big \
        file=newstest2014.en \
        vocab_file=vocab.ende.32768 \
        file_out=translation.en \
        reference=newstest2014.de \
        warmup_steps=3 \
        steps=100 
```
#### Accuracy mode:
```
  python3 ./benchmarks/launch_benchmark.py    \
     --accuracy-only --framework tensorflow  \
     --in-graph=$PB_FILE \
     --model-name transformer_mlperf \
     --mode inference --precision int8 \
     --batch-size $BATCH_SIZE \
     --num-intra-threads $NUM_CORES --num-inter-threads $NUM_SOCKETS \
     --verbose \
     --data-location $DATA_DIR \
     --docker-image intel/intel-optimized-tensorflow:latest \
     -- params=big \
        file=newstest2014.en \
        vocab_file=vocab.ende.32768 \
        file_out=translation.en \
        reference=newstest2014.de \
        steps=100 
```
where:
   * $DATA_DIR -- the input data directory, which should include newstest2014.en, newstest2014.de and vocab.ende.32768
   * $PB_FILE  -- the path of the quantized model downloaded from the trained models website the Intel published. 
   * steps -- the number of batches of data to feed into the model for inference, if the number is greater than available batches in the input data, it will only run number of batches available in the data.

The log file is saved to the value of --output-dir. if not value spacified, the log will be at the models/benchmarks/common/tensorflow/logs in workspace.
With accuracy mode, the official BLEU score will be printed

The performance and accuracy in the the log output when the benchmarking completes should look
something like this, the real throughput and inferencing time varies:
```
  Total inferencing time: xxx
  Throughput: xxx  sentences/second
  Case-insensitive results: 26.664000749588013
  Case-sensitive results: 26.154428720474243

```

## Run the model on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/tensorflow/Windows.md).

Before running inference, users should have the model fully trained and have saved checkpoints ready at the path `%CHECKPOINT_DIR%`.
In order to improve the performance, we added a new script to generate a frozen model from a fully trained model checkpoint.
To generate the frozen model, users need to run the following command in the transformer model directory where [export_transformer.py](/models/language_translation/tensorflow/transformer_mlperf/inference/int8/transformer/export_transformer.py) in.

Using `cmd.exe`, run:
```
set PYTHONPATH=%PYTHONPATH%;<PATH_TO_MODEL_ZOO_ROOT>\models\common\tensorflow
python export_transformer.py --model_dir=<%CHECKPOINT_DIR%> --pb_path=<frozen_graph_full_path>
```
The translate can be run with accuracy mode or benchmark mode. The benchmark mode will run with the best performance by setting warmup steps and the total steps users want to run.
The accuracy mode will just run for testing accuracy without setting warmup steps and steps.

Set the environment variables to point to the dataset directory `DATA_DIR`, the path to the pretrained model file `PB_FILE`, batch size `BATCH_SIZE`, and  the number of sockets `NUM_SOCKETS`.
You can use `wmic cpu get SocketDesignation` to list the available socket on your system, then set `NUM_SOCKETS` accordingly.
```
set PB_FILE=<path to the directory where the frozen pre trained model file saved>\\transformer_mlperf_int8.pb
set DATA_DIR=<the input data directory, which should include newstest2014.en, newstest2014.de and vocab.ende.32768>
set BATCH_SIZE=1
set NUM_SOCKETS=2
```
#### Benchmark mode:
Using `cmd.exe`, run:
```
  cd benchmarks
  python launch_benchmark.py    ^
     --benchmark-only --framework tensorflow  ^
     --in-graph=%PB_FILE% ^
     --model-name transformer_mlperf ^
     --mode inference --precision int8 ^
     --batch-size %BATCH_SIZE% ^
     --num-intra-threads %NUMBER_OF_PROCESSORS% --num-inter-threads %NUM_SOCKETS% ^
     --verbose ^
     --data-location %DATA_DIR% ^
     -- params=big ^
        file=newstest2014.en ^
        vocab_file=vocab.ende.32768 ^
        file_out=translation.en ^
        reference=newstest2014.de ^
        warmup_steps=3 ^
        steps=100 
```
#### Accuracy mode:
Using `cmd.exe`, run:
```
  cd benchmarks
  python launch_benchmark.py    ^
     --accuracy-only --framework tensorflow  ^
     --in-graph=%PB_FILE% ^
     --model-name transformer_mlperf ^
     --mode inference --precision int8 ^
     --batch-size %BATCH_SIZE% ^
     --num-intra-threads %NUMBER_OF_PROCESSORS% --num-inter-threads %NUM_SOCKETS% ^
     --verbose ^
     --data-location %DATA_DIR% ^
     -- params=big ^
        file=newstest2014.en ^
        vocab_file=vocab.ende.32768 ^
        file_out=translation.en ^
        reference=newstest2014.de ^
        steps=100 
```
where:
   * steps -- the number of batches of data to feed into the model for inference, if the number is greater than available batches in the input data, it will only run number of batches available in the data.

