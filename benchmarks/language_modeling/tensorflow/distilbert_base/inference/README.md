<!--- 0. Title -->
# DistilBERT inference

<!-- 10. Description -->

### Description
This document has instructions for running distilBERT inference for FP32 and BF16. Inference is performed for the task of text classification (binary) on full english sentences from the below mentioned dataset. The distilbert model based on this [paper](https://arxiv.org/abs/1910.01108).
The [pretrained-model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you) thus used, was taken from [Hugging face model repository](https://huggingface.co/models).

### Dataset
We use a part of Stanford Sentiment Treebank corpus for our task. Specifically, the validation split present in the SST2 dataset in the hugging face [repository](https://huggingface.co/datasets/sst2). It contains 872 labeled English sentences. Instructions to get the frozen graph and dataset are given below:

```
Download Frozen graph (FP32 - The FP32 frozen graph should used for FP32, BFloat16 and FP16):
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/distilbert_frozen_graph_fp32_final.pb

Download Frozen graph for INT8:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/FusedBMM_int8_distilbert_frozen_graph.pb

Download Frozen graph for IN8 with OneDNN graph (Only used when the plugin Intel Extension for Tensorflow is installed, as OneDNN Graph optimization is enabled by default at this point):
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_12_0/distilbert_itex_int8.pb

Downloading dataset:
cd to directory:  <model_zoo_dir>/models/language_modeling/tensorflow/distilbert_base/inference/
python download_dataset.py --path_to_save_dataset <enter path to save dataset>

```

## Run the model

### Install requirements

```
cd to the directory: <model_zoo_dir>/models/language_modeling/tensorflow/distilbert_base/inference
pip install -r requirements.txt

Set the required Environment variables:

export BATCH_SIZE=<set to required batch size>
export IN_GRAPH=<path to downloaded frozen graph>
export PRECISION=<set precision fp32, fp16, bfloat16 or int8>
export DATASET_DIR=<path to the downloaded dataset directory>
export WARMUP_STEPS=<set to required warmup steps>
export OUTPUT_DIR=<directory to store log files of the run>
```

Use quickstart scripts for running the model

```
cd <model_dir>
```
Accuracy

```
./quickstart/language_modeling/tensorflow/distilbert_base/inference/cpu/accuracy.sh
```

Throuhgput:

(runs with a batch size of 56)

```
./quickstart/language_modeling/tensorflow/distilbert_base/inference/cpu/inference_realtime_multi_instance.sh
```

Latency:

(runs with a batch size of 1)

```
./quickstart/language_modeling/tensorflow/distilbert_base/inference/cpu/inference_throughput_multi_instance.sh
```

Weight-sharing:

```
./quickstart/language_modeling/tensorflow/distilbert_base/inference/cpu/inference_realtime_weight_sharing.sh
```
