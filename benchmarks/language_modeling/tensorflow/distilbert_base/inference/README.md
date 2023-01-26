<!--- 0. Title -->
# DistilBERT inference

<!-- 10. Description -->

### Description
This document has instructions for running distilBERT inference for FP32 and BF16. Inference is performed for the task of text classification (binary) on full english sentences from the below mentioned dataset. The distilbert model based on this [paper](https://arxiv.org/abs/1910.01108).
The [pretrained-model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you) thus used, was taken from [Hugging face model repository](https://huggingface.co/models).

### Dataset
We use a part of Stanford Sentiment Treebank corpus for our task. Specifically, the validation split present in the SST2 dataset in the hugging face [repository](https://huggingface.co/datasets/sst2). It contains 872 labeled English sentences. Instructions to get the frozen graph and dataset are given below:

```
Download Frozen graph:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/distilbert_frozen_graph_fp32_final.pb

Download Frozen graph for INT8:
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/FusedBMM_int8_distilbert_frozen_graph.pb

Downloading dataset:
cd to directory:  <model_zoo_dir>/models/language_modeling/tensorflow/distilbert_base/inference/
python download_dataset.py --path_to_save_dataset <enter path to save dataset>

```

## Run the model

### Install requirements

```
cd to the directory: <model_zoo_dir>/models/language_modeling/tensorflow/distilbert_base/inference
pip install -r requirements.txt
```

### Run command

Use <model_zoo_dir>/benchmarks/launch_benchmark.py to run inference for distilbert

FP32:

```
python launch_benchmark.py 
--model_name distilbert_base \
--mode inference \
--framework tensorflow \
--precision fp32 \
--batch_size 32 \ 
--benchmark-only \
--in-graph <path to frozen graph (.pb)> \
--data-location <path to the saved dataset> \
--warmup-steps=20
```

BFLOAT16:

```
python launch_benchmark.py 
--model_name distilbert_base \
--mode inference \
--framework tensorflow \
--precision bfloat16 \
--batch_size 32 \ 
--benchmark-only \
--in-graph <path to frozen graph (.pb)>
--data-location <path to the saved dataset>
--warmup-steps=20
```

INT8:

```
python launch_benchmark.py 
--model_name distilbert_base \
--mode inference \
--framework tensorflow \
--precision int8 \
--batch_size 32 \ 
--benchmark-only \
--in-graph [path to frozen graph (.pb)]
--data-location [path to the saved dataset]
--warmup-steps=20
```

Other options: \
`--accuracy-only` to get accuracy as well
`--max-seq-length=128` uses 128 by default
