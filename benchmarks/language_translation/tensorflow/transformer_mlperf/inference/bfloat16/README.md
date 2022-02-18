<!--- 0. Title -->
# Transformer Language BFLOAT16 Inference

<!-- 10. Description -->
## Description

This document has instructions for running Transformer Language FP32 Inference in mlperf
Benchmark suits using Intel-optimized TensorFlow.

Detailed information on mlperf Benchmark can be found in [mlperf/training](https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)

The inference code is based on the transformer mlperf evaluation code, but Intel has optimized the inference model by modify the code of the model, so that it can achieve better performance on Intel CPUs. 
The bfloat16 model is manually modfied by cast fp32 data type tensor to bfloat16 in the model, and we trained the modified model to reach the same or higher accuracy. The inference is based on the bfloat16 model we trained.

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATA_DIR` to point out to the location of the dataset directory.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

Transformer Language in mlperf benchmark can run with full training or
fewer training steps. During training we can control if it will do the evaluation
or not.

## Run the model

Before running inference, users should have the model fully trained and have saved checkpoints ready at the path $CHECKPOINT_DIR.
In order to improve the performance, we added a new script to generate a frozen model from a fully trained model checkpoint.
To generate the frozen model, users need to run the following command in the tranformer model directory where export_transformer.py in:

```
export PYTHONPATH=$PYTHONPATH:<PATH_TO_MODEL_ZOO_ROOT>/models/common/tensorflow

python export_transformer.py --model_dir=<$CHECKPOINT_DIR> --pb_path=<frozen_graph_full_path>
```
The translate can be run with accuracy mode or benchmark mode. The benchmark mode will run with the best performance by setting warmup steps and the total steps users want to run. The accuracy mode will just run for testing accuracy without setting warmup steps and steps.

#### Benchmark mode run:
```
  python3 ./benchmarks/launch_benchmark.py    \
     --benchmark-only --framework tensorflow  \
     --in-graph=$PB_FILE \
     --model-name transformer_mlperf \
     --mode inference --precision bfloat16\
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
#### accuracy mode run:
```
  python3 ./benchmarks/launch_benchmark.py    \
     --accuracy-only --framework tensorflow  \
     --in-graph=$PB_FILE \
     --model-name transformer_mlperf \
     --mode inference --precision bfloat16 \
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
   * $PB_FILE  -- the path of the frozen model generated with the script
   * steps -- the number of batches of data to feed into the model for inference, if the number is greater than avaialable batches in the input data, it will only run number of batches available in the data.

The log file is saved to the value of --output-dir. if not value spacified, the log will be at the models/benchmarks/common/tensorflow/logs in workspace.
With accuracy mode, the official BLEU score will be printed

The performance and accuracy in the the log output when the benchmarking completes should look
something like this, the real throughput and inferencing time varies:
```
  Total inferencing time: xxx
  Throughput: xxx  sentences/second
  Case-insensitive results: 27.636119723320007
  Case-sensitive results: 27.127626538276672

