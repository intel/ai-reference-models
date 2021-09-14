<!--- 30. Datasets -->
## Datasets

### SQuAD data
Download and unzip the <model name> uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Set the `DATASET_DIR` to point to this directory when running <model name>.
```
mkdir -p $DATASET_DIR && cd $DATASET_DIR
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```

Follow [instructions to generate BERT pre-training dataset](https://github.com/IntelAI/models/blob/bert-lamb-pretraining-tf-2.2/quickstart/language_modeling/tensorflow/bert_large/training/bfloat16/HowToGenerateBERTPretrainingDataset.txt)
in TensorFlow record file format. The output TensorFlow record files are expected to be located in the dataset directory `${DATASET_DIR}/tf_records`. An example for the TF record file path should be
`${DATASET_DIR}/tf_records/part-00430-of-00500`.
