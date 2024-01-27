# Bert Large Inference

BERT Large Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert        |           -           |         -          |

# Pre-Requisite
* Host has one of the following GPUs:
  * **Arc Series** - [Intel® Arc™ A-Series Graphics](https://ark.intel.com/content/www/us/en/ark/products/series/227957/intel-arc-a-series-graphics.html)
  * **Max Series** - [Intel® Data Center GPU Max Series](https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html)
* Host has installed latest Intel® Data Center GPU Max & ARC Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library
  - Intel® oneAPI CCL Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.


## Datasets

### SQuAD dataset

Download the [SQuAD 1.0 dataset](https://github.com/huggingface/transformers/tree/v4.0.0/examples/question-answering#fine-tuning-bert-on-squad10).
Set the `DATASET_DIR` to point to the directory where the files are located before running the BERT quickstart scripts. Your dataset directory should look something
like this:
```bash
<DATASET_DIR>/
├── dev-v1.1.json
├── evaluate-v1.1.py
└── train-v1.1.json
```
The setup assumes the dataset is downloaded to the current directory. 

## Pre-trained Model

Download the `config.json` and fine tuned model from huggingface and set the `BERT_WEIGHT` environment variable to point to the directory that has both files:

```bash
mkdir squad_large_finetuned_checkpoint
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json -O squad_large_finetuned_checkpoint/config.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin  -O squad_large_finetuned_checkpoint/pytorch_model.bin
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json -O squad_large_finetuned_checkpoint/tokenizer.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer_config.json -O squad_large_finetuned_checkpoint/tokenizer_config.json
wget -c https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt -O squad_large_finetuned_checkpoint/vocab.txt

BERT_WEIGHT=<path_to_BERT_WEIGHT_directory>/squad_large_finetuned_checkpoint
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/bert_large/inference/gpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest GPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation):
    ```
    python -m pip install torch==<torch_version> torchvision==<torchvvision_version> intel-extension-for-pytorch==<ipex_version> --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
    ```
6. Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
    ```
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=Max` (Max or ARC)                                                 |
| **BERT_WEIGHT**              | `export BERT_WEIGHT=<path_to_BERT_WEIGHT_directory>/squad_large_finetuned_checkpoint` |
| **DATASET_DIR**              | `export DATASET_DIR=<path/to/dataset>`                  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |
| **PRECISION** (optional)     |                  `export PRECISION=BF16` (BF16, FP32 and FP16 are supported for Max and FP16 for ARC) |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=-1`                             |
8. Run `run_model.sh`

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
