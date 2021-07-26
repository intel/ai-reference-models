# Transformer Language

The following documents have instructions for how to run Transformer Language used in mlperf
Benchmark suites for the following modes/platforms:
* [FP32 training](/benchmarks/language_translation/tensorflow/transformer_mlperf/training/fp32/README.md)
* [Bfloat16 training](/benchmarks/language_translation/tensorflow/transformer_mlperf/training/bfloat16/README.md)
* [FP32 inference](#fp32-inference-instructions)
* [Bfloat16 inference](#bfloat16-inference-instructions)

Detailed information on Benchmark can be found in [mlperf/training](https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)

# <a name="fp32-inference-instructions"></a> FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
git clone https://github.com/IntelAI/models.git
```

2. Obtain the dataset.
Decide the problem you want to run to get the appropriate dataset.
We will get the training data of it as an example:

    Download dataset for computing BLEU score reported in the paper
    ```
    export DATA_DIR=/home/<user>/transformer_data
    mkdir $DATA_DIR && cd $DATA_DIR
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
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
    --docker-image intel/intel-optimized-tensorflow:latest \
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
git clone https://github.com/IntelAI/models.git
```

2. Obtain the dataset.
Decide the problem you want to run to get the appropriate dataset.
We will get the training data of it as an example:

    Download dataset for computing BLEU score reported in the paper
    ```
    export DATA_DIR=/home/<user>/transformer_data
    mkdir $DATA_DIR && cd $DATA_DIR
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
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
    --docker-image intel/intel-optimized-tensorflow:latest \
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


