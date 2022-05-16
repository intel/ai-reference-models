# BERT Large datasets

## Inference
Download and unzip the BERT Large uncased (whole word masking) model from the
[Google BERT repository](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD 1.1) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```

## Training
For both fine-tuning and classification training, the datasets (SQuAD, MultiNLI, MRPC etc..) and checkpoints
will be downloaded based on the [Google BERT repository](https://github.com/google-research/bert).

### Fine-Tuning with BERT using SQuAD data.
1. Download and extract one of BERT large pretrained models from [Google BERT repository](https://github.com/google-research/bert#pre-trained-models).
As an example, you can use the BERT Large uncased (whole word masking) model.
The extracted directory should be set to the `CHECKPOINT_DIR` environment
variable when running the quickstart scripts.
```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip
```

2. Download the Stanford Question Answering Dataset (SQuAD) dataset files from the [Google BERT repository](https://github.com/google-research/bert#squad-11).
The three files (`train-v1.1.json`, `dev-v1.1.json`, and `evaluate-v1.1.py`)
should be downloaded to the same directory. Set the `DATASET_DIR` to point to
that directory when running bert fine tuning using the SQuAD data.

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget evaluate-v1.1.py https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
```

### Classification Training with BERT
1. Download and extract the BERT base uncased 12-layer, 768-hidden pretrained model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
The extracted directory should be set to the `CHECKPOINT_DIR` environment
variable when running the quickstart scripts.
```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

2. For training from scratch Microsoft Research Paraphrase Corpus (MRPC) data needs to be downloaded and pre-processed.
Download the MRPC data from [General Language Understanding Evaluation (GLUE) Data](https://gluebenchmark.com/tasks), and save it to some directory directory `${DATASET_DIR}`.
Set the `DATASET_DIR` to point to that directory.
You can also use the helper script [download_glue_data.py](https://gist.github.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3) to download the data:
```
# Download a copy of download_glue_data.py to the current directory
wget https://gist.githubusercontent.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3/raw/db67cdf22eb5bd7efe376205e8a95028942e263d/download_glue_data.py
python3 download_glue_data.py --data_dir ${DATASET_DIR} --tasks MRPC
```
