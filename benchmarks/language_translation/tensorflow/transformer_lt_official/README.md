# Transformer Language Translation (LT) Official

This document has instructions for how to run Transformer Language official from TensorFlow models
for the following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model inference for other platforms are coming later.

## FP32 Inference Instructions

1. Download and extract the frozen graph of the model and necessary data files.

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/transformer_lt_official_fp32_pretrained_model.tar.gz
$ tar -xzvf transformer_lt_official_fp32_pretrained_model.tar.gz
$ ls -l transformer_lt_official_fp32_pretrained_model/*
transformer_lt_official_fp32_pretrained_model/graph:
total 241540
-rwx------. 1 user group 247333269 Mar 15 17:29 fp32_graphdef.pb

transformer_lt_official_fp32_pretrained_model/data:
total 1064
-rw-r--r--. 1 user group 359898 Feb 20 16:05 newstest2014.en
-rw-r--r--. 1 user group 399406 Feb 20 16:05 newstest2014.de
-rw-r--r--. 1 user group 324025 Mar 15 17:31 vocab.txt
```

2. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

3. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo (from step 3).
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the dataset location (from step 2).

Transformer LT official can run for online or batch inference. Use one of the following examples below, depending on
your use case.

For online inference (using `--socket-id 0` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name transformer_lt_official \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --in-graph /home/<user>/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb \
    --data-location /home/<user>/transformer_lt_official_fp32_pretrained_model/data \
    -- file=newstest2014.en \
    file_out=translate.txt \
    reference=newstest2014.de \
    vocab_file=vocab.txt
```

For batch inference (using `--socket-id 0` and `--batch-size 64`):

```
python launch_benchmark.py \
    --model-name transformer_lt_official \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 64 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0 \
    --in-graph /home/<user>/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb \
    --data-location /home/<user>/transformer_lt_official_fp32_pretrained_model/data \
    -- file=newstest2014.en \
    file_out=translate.txt \
    reference=newstest2014.de \
    vocab_file=vocab.txt

```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.
The num-inter-threads and num-intra-threads could be set different numbers depends on 
the CPU in the system to achieve the best performance.

4.  The log file and default translated results is saved to the `models/benchmarks/common/tensorflow/logs` directory.

