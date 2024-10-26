# ViT Inference

Vision Transformer inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://huggingface.co/google/vit-base-patch16-224        |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#installation)

## Bare Metal
### Model Specific Setup

* Install Intel OpenMP
  ```
  pip install packaging intel-openmp accelerate==0.34.1
  ```
* Set IOMP, jemalloc and tcmalloc Preload for better performance
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"<path_to>/tcmalloc/lib/libtcmalloc.so":"<path_to_iomp>/lib/libiomp5.so":$LD_PRELOAD
  ```

* Install datasets
  ```
  pip install datasets
  ```

* Set CORE_PER_INSTANCE before running realtime mode
  ```
  export CORE_PER_INSTANCE=4
  (4cores per instance setting is preferred, while you could set any other config like 1core per instance)
  ```

* About the BATCH_SIZE in scripts
  ```
  Throughput mode is using BATCH_SIZE=[4 x core number] by default in script (which could be further tuned according to the testing host);
  Realtime mode is using BATCH_SIZE=[1] by default in script;
  ```

* Do calibration to get quantization config before running INT8.
  ```
  bash do_calibration.sh
  ```

* [optional] you may need to get access to llama2 weights from HF
  Apply the access in the pages with your huggingface account:
  - LLaMA2 7B : https://huggingface.co/meta-llama/Llama-2-7b-hf
  - LLaMA2 13B : https://huggingface.co/meta-llama/Llama-2-13b-hf

  huggingface-cli login
  {your huggingface token}

* [Optional] Use dummy input for performance collection
  ```
  export DUMMY_INPUT=1
  ```

# Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/vit/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)

6. Prepare for downloading access
    On https://huggingface.co/datasets/imagenet-1k, login your account, and click the aggreement and then generating {your huggingface token}

    huggingface-cli login
    {your huggingface token}

7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$(pwd)`                               |
| **DATASET_DIR**          |  `export DATASET_DIR=<path to dataset dir>`    |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, bf16, fp16, int8-fp32, int8-bf16) |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |
| **DUMMY_INPUT**(optional)     |     `export DUMMY_INPUT=1` (This is optional; for performance collection)    |
| **CORE_PER_INSTANCE** (required for REALTIME)    |                               `export CORE_PER_INSTANCE=4`                                |
7. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
2023-11-15 06:22:47,398 - __main__ - INFO - Results: {'exact': 87.01040681173131, 'f1': 93.17865304772475, 'total': 10570, 'HasAns_exact': 87.01040681173131, 'HasAns_f1': 93.17865304772475, 'HasAns_total': 10570, 'best_exact': 87.01040681173131, 'best_exact_thresh': 0.0, 'best_f1': 93.17865304772475, 'best_f1_thresh': 0.0}
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 405.9567
   unit: example/s
 - key: latency
   value: 0.15765228112538657
   unit: s/example
 - key: accuracy
   value: 93.179
   unit: f1
```
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
