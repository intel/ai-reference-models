# CHATGLMv3 6B Inference

CHATGLMv3 6B inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://huggingface.co/THUDM/chatglm3-6b       |           -           |         -          |

# Pre-Requisite
## Bare Metal
### General setup

Follow [link](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md) to build Pytorch, IPEX, TorchVison and TCMalloc.

### Model Specific Setup

* Install Intel OpenMP
  ```
  pip install packaging intel-openmp accelerate
  ```
* Set IOMP and tcmalloc Preload for better performance
  ```
  export LD_PRELOAD="<path_to>/tcmalloc/lib/libtcmalloc.so":"<path_to_iomp>/lib/libiomp5.so":$LD_PRELOAD
  ```

* Set ENV to use fp16 AMX if you are using a supported platform
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
  ```

# Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/chatglm/inference/cpu`
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

6.  [Optional] Specify a specific commit version of the model
    ```
    export REVISION=9addbe01105ca1939dd60a0e5866a1812be9daea
    ```

  * About the BATCH_SIZE in scripts
    ```
    using BATCH_SIZE=1 for realtime mode
    using BATCH_SIZE=N for throughput mode (N could be further tuned according to the testing host, by default using 1);
    ```

  * About the BEAM_SIZE in scripts
    ```
    using BEAM_SIZE=4 by default
    ```

  * Do calibration to get "qconfig.json" before running INT8.
    ```
    # You can get "qconfig.json" for calibration:
    bash do_quantization.sh calibration sq #using smooth quant as default
    ```

7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$(pwd)`                               |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, bf16, fp16, int8) |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **INPUT_TOKEN** | `export INPUT_TOKEN=32(choice in [32 64 128 256 512 1024 2016], we prefer to benchmark on 32 and 2016)`   |
| **OUTPUT_TOKEN** | `export OUTPUT_TOKEN=32(32 is preferred, while you could set any other length)`   |
| **BATCH_SIZE** (optional)   |                               `export BATCH_SIZE=256`                                |

7. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
 ---------- Summary: ----------
inference-latency: 168.207 sec.
first-token-latency: 38.174 sec.
rest-token-latency: 4.188 sec.
P90-rest-token-latency: 4.210 sec.
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
- key: first token latency
  value: 38.174
- key: rest_token_latency
  value: 4.188
- key: accuracy
  value: 93.17
```
