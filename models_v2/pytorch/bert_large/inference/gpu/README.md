# Bert Large Inference

BERT Large Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU Max & ARC
* Host has installed latest Intel® Data Center GPU Max & ARC Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Prepare Dataset
please follow below command to download checkpoints and dataset. 

if ! test -d ./SQUAD1;then mkdir -p SQUAD1 && cd SQUAD1 && wget https://data.deepai.org/squad1.1.zip && unzip squad1.1.zip;cd ../;fi

if ! test -d ./squad_large_finetuned_checkpoint;then mkdir squad_large_finetuned_checkpoint && cd squad_large_finetuned_checkpoint && wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json && wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin && wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json && wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer_config.json && wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt;cd ../;fi


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/bert_large/inference/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=PVC` (PVC or ARC)                                                 |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |
| **PRECISION** (optional)     |                  `export PRECISION=BF16` (BF16, FP32 and FP16 are supported for PVC and FP16 for ARC) |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=-1`                             |
6. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
2023-11-15 06:22:47,398 - __main__ - INFO - Results: {'exact': 87.01040681173131, 'f1': 93.17865304772475, 'total': 10570, 'HasAns_exact': 87.01040681173131, 'HasAns_f1': 93.17865304772475, 'HasAns_total': 10570, 'best_exact': 87.01040681173131, 'best_exact_thresh': 0.0, 'best_f1': 93.17865304772475, 'best_f1_thresh': 0.0}
```

Multi-tile output will typically looks like:
```
2023-11-15 06:29:34,737 - __main__ - INFO - Results: {'exact': 87.01040681173131, 'f1': 93.17865304772475, 'total': 10570, 'HasAns_exact': 87.01040681173131, 'HasAns_f1': 93.17865304772475, 'HasAns_total': 10570, 'best_exact': 87.01040681173131, 'best_exact_thresh': 0.0, 'best_f1': 93.17865304772475, 'best_f1_thresh': 0.0}
2023-11-15 06:29:35,599 - __main__ - INFO - Results: {'exact': 87.01040681173131, 'f1': 93.17865304772475, 'total': 10570, 'HasAns_exact': 87.01040681173131, 'HasAns_f1': 93.17865304772475, 'HasAns_total': 10570, 'best_exact': 87.01040681173131, 'best_exact_thresh': 0.0, 'best_f1': 93.17865304772475, 'best_f1_thresh': 0.0}
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 405.9567
   unit: sent/s
 - key: latency
   value: 0.15765228112538657
   unit: s
 - key: accuracy
   value: 93.179
   unit: f1
```
