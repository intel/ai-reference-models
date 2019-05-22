# Transformer Language

This document has instructions for how to run Transformer Language benchmark for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference for
other platforms are coming later.

## FP32 Inference Instructions

1. Clone an older commit from the [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) repository:

```
$ git clone https://github.com/tensorflow/tensor2tensor.git
$ cd tensor2tensor
$ git checkout 6cea4c460585ce835d8bfa87f7191e29dd4de9e2
```

2. Obtain the dataset
Decide the problem you want to run to get the appropriate dataset. In our default case, we are running translate_ende_wmt32k.
We will get the training data of it as an example:

```
$ PYTHONPATH=$PYTHONPATH:$RootDirOfTensor2tensor \
  ./tensor2tensor/bin/t2t-datagen \
    --problem=translate_ende_wmt32k \
    --data_dir=/home/<user>/t2t_data \
    --tmp_dir=~/home/<user>/t2t_data/tmp
```

3. Download and extract the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/transformer_lt_fp32_pretrained_model.tar.gz
$ tar -xzvf transformer_lt_fp32_pretrained_model.tar.gz
$ ls -l transformer_lt_fp32_pretrained_model
total 750000
-rw-r--r-- 1 <user> <group>      1003 Dec  7 15:20 checkpoint
-rw-r--r-- 1 <user> <group>       939 Dec  7 15:20 flags_t2t.txt
-rw-r--r-- 1 <user> <group>      1432 Dec  7 15:20 flags.txt
-rw-r--r-- 1 <user> <group>  19752171 Dec  7 15:20 graph.pbtxt
-rw-r--r-- 1 <user> <group>      3218 Dec  7 15:20 hparams.json
-rw-r--r-- 1 <user> <group>        24 Dec  7 15:20 model.ckpt-340000.data-00000-of-00002
-rw-r--r-- 1 <user> <group> 736542728 Dec  7 15:20 model.ckpt-340000.data-00001-of-00002
-rw-r--r-- 1 <user> <group>     29961 Dec  7 15:20 model.ckpt-340000.index
-rw-r--r-- 1 <user> <group>  11127562 Dec  7 15:20 model.ckpt-340000.meta
-rw-r--r-- 1 <user> <group>    268951 Dec  7 15:20 newstest2015.de
-rw-r--r-- 1 <user> <group>    241016 Dec  7 15:20 newstest2015.en
```

4. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

5. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo (from step 4).
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a benchmarking run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image to use, along with your path to the dataset location (from step 2),
and the checkpoint directory (from step 3).

Substitute the `--model-source-dir` for the location where you cloned the
[tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) repo
(from step 1).

Transformer Language can run for latency or throughput
benchmarking. Use one of the following examples below, depending on
your use case. Note that if no `reference` file is provided in the
launch script parameters, then the BLEU score cannot be calculated.

For latency (using `--socket-id 0` and `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name transformer_language \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:1.14 \
    --checkpoint /home/<user>/transformer_lt_fp32_pretrained_model \
    --data-location /home/<user>/t2t_data \
    --model-source-dir /home/<user>/tensor2tensor/ \
    -- decode_from_file=newstest2015.en reference=newstest2015.de
```

For throughput (using `--socket-id 0` and `--batch-size 32`):

```
python launch_benchmark.py \
    --model-name transformer_language \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 32 \
    --socket-id 0 \
    --docker-image intelaipg/intel-optimized-tensorflow:1.14 \
    --checkpoint /home/<user>/transformer_lt_fp32_pretrained_model \
    --data-location /home/<user>/t2t_data \
    --model-source-dir /home/<user>/tensor2tensor/ \
    -- decode_from_file=newstest2015.en reference=newstest2015.de
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

6.  The log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when benchmarking for latency:
```
INFO:tensorflow:Decoding batch 2167
INFO:tensorflow:Inference results INPUT: Move!
INFO:tensorflow:Inference results OUTPUT: Move!
INFO:tensorflow:Decoding batch 2168
INFO:tensorflow:Inference results INPUT: Fantastic.
INFO:tensorflow:Inference results OUTPUT: Fantastisch.
INFO:tensorflow:Writing decodes into /workspace/models/out_dir/output_infer
  Inference time 6094.9205, Latency = 2810.0141 ms/setences
BLEU_uncased =  22.63
BLEU_cased =  22.20
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_transformer_language_inference_fp32_20190210_050451.log
```

Example log tail when benchmarking for throughput:
```
INFO:tensorflow:Inference results INPUT: Move!
INFO:tensorflow:Inference results OUTPUT: Move!
INFO:tensorflow:Inference results INPUT: Fantastic.
INFO:tensorflow:Inference results OUTPUT: Fantastisch.
INFO:tensorflow:Writing decodes into /workspace/models/out_dir/output_infer
  Inference time 1174.0522, Throughput = 1.8474 sentences/second
BLEU_uncased =  22.63
BLEU_cased =  22.20
Ran inference with batch size 32
Log location outside container: {--output-dir value}/benchmark_transformer_language_inference_fp32_20190210_072635.log
```