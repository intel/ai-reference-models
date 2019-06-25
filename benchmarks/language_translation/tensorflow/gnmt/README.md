# GNMT

This document has instructions for how to run GNMT for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions

1. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

This repository includes launch scripts for running 
an optimized version of the GNMT model code.

2. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/gnmt_4layer_fp32_pretrained_model.tar.gz
```

3. To run GNMT inference, you will WMT16 German-English data. You can
download the dataset using the script provided on nmt github.

```
$git clone https://github.com/tensorflow/nmt.git
Cloning into 'nmt'...
remote: Enumerating objects: 1247, done.
remote: Total 1247 (delta 0), reused 0 (delta 0), pack-reused 1247
Receiving objects: 100% (1247/1247), 1.23 MiB | 7.72 MiB/s, done.
Resolving deltas: 100% (891/891), done.

$nmt/scripts/wmt16_en_de.sh /home/<user>/wmt16
```

After the script has completed, you should have a directory with the
dataset looks like:

```
$ ls /home/<user>/wmt16/
bpe.32000                      newstest2010.tok.de            newstest2012.tok.en            newstest2015.de                train.de
data                           newstest2010.tok.en            newstest2013.de                newstest2015.en                train.en
mosesdecoder                   newstest2011.de                newstest2013.en                newstest2015.tok.bpe.32000.de  train.tok.bpe.32000.de
newstest2009.de                newstest2011.en                newstest2013.tok.bpe.32000.de  newstest2015.tok.bpe.32000.en  train.tok.bpe.32000.en
newstest2009.en                newstest2011.tok.bpe.32000.de  newstest2013.tok.bpe.32000.en  newstest2015.tok.de            train.tok.clean.bpe.32000.de
newstest2009.tok.bpe.32000.de  newstest2011.tok.bpe.32000.en  newstest2013.tok.de            newstest2015.tok.en            train.tok.clean.bpe.32000.en
newstest2009.tok.bpe.32000.en  newstest2011.tok.de            newstest2013.tok.en            newstest2016.de                train.tok.clean.de
newstest2009.tok.de            newstest2011.tok.en            newstest2014.de                newstest2016.en                train.tok.clean.en
newstest2009.tok.en            newstest2012.de                newstest2014.en                newstest2016.tok.bpe.32000.de  train.tok.de
newstest2010.de                newstest2012.en                newstest2014.tok.bpe.32000.de  newstest2016.tok.bpe.32000.en  train.tok.en
newstest2010.en                newstest2012.tok.bpe.32000.de  newstest2014.tok.bpe.32000.en  newstest2016.tok.de            vocab.bpe.32000
newstest2010.tok.bpe.32000.de  newstest2012.tok.bpe.32000.en  newstest2014.tok.de            newstest2016.tok.en            vocab.bpe.32000.de
newstest2010.tok.bpe.32000.en  newstest2012.tok.de            newstest2014.tok.en            subword-nmt                    vocab.bpe.32000.en
```

4. Next, navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
platform, and docker image to use, along with your path to the dataset
that you generated in step 3.

Substitute in your own `--data-location` (from step 3), `--checkpoint` pre-trained
model file path (from step 2), and the name/tag for your docker image.

GNMT can be run for online and batch inference. Use one of
the following examples below, depending on your use case.

For online inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):

```
python launch_benchmark.py \
--model-name gnmt \
--precision fp32 \
--mode inference \
--framework tensorflow \
--benchmark-only \
--batch-size 1 \
--socket-id 0 \
--checkpoint /home/<user>/gnmt_checkpoints \
--data-location /home/<user>/wmt16 \
--docker-image intelaipg/intel-optimized-tensorflow:1.14.0 \
-- infer_mode=beam_search
```

For batch inference (using `--benchmark-only`, `--socket-id 0` and `--batch-size 32`):

```
python launch_benchmark.py \
--model-name gnmt \
--precision fp32 \
--mode inference \
--framework tensorflow \
--benchmark-only \
--batch-size 32 \
--socket-id 0 \
--checkpoint /home/<user>/gnmt_checkpoints \
--data-location /home/<user>/wmt16 \
--docker-image intelaipg/intel-optimized-tensorflow:1.14.0 \
-- infer_mode=beam_search
```

6. The log file is saved to the
`models/benchmarks/common/tensorflow/logs` directory. Below are
examples of what the tail of your log file should look like for the
different configs.

Example log tail when running for online inference:
```
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_3/basic_lstm_cell/bias:0, (4096,), /device:CPU:0
  dynamic_seq2seq/decoder/output_projection/kernel:0, (1024, 36548),
  loaded infer model parameters from /checkpoints/translate.ckpt, time 1.09s
# Start decoding
  decoding to output /workspace/benchmarks/language_translation/tensorflow/gnmt/inference/fp32/out_dir/output_infer
  done, num sentences 2169, num translations per input 1, time 1108s, Wed Feb  6 01:36:13 2019.
  The latency of the model is 511.2466 ms/sentences
  bleu: 29.2
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_gnmt_inference_fp32_20190206_011740.log
```

Example log tail when running for batch inference:
```
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_3/basic_lstm_cell/bias:0, (4096,), /device:CPU:0
  dynamic_seq2seq/decoder/output_projection/kernel:0, (1024, 36548),
  loaded infer model parameters from /checkpoints/translate.ckpt, time 1.08s
# Start decoding
  decoding to output /workspace/benchmarks/language_translation/tensorflow/gnmt/inference/fp32/out_dir/output_infer
  done, num sentences 2169, num translations per input 1, time 302s, Wed Feb  6 01:48:30 2019.
  The throughput of the model is 7.1780 sentences/s
  bleu: 29.2
Ran inference with batch size 32
Log location outside container: {--output-dir value}/benchmark_gnmt_inference_fp32_20190206_014324.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..