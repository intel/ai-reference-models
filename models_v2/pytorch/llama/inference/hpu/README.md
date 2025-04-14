<!--- 0. Title -->
# PyTorch LLaMA2 and LLaMA38B inference using Optimum-Habana and vLLM on Gaudi (generation)

<!-- 10. Description -->
## Description

This document has instructions for running Llama2 and Llama3 inference (generation) using Optimum-Habana and vLLM on Gaudi.

## Bare Metal
### General setup
Please make sure to follow [Driver Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install Gaudi driver on the system.
To use dockerfile provided for the sample, please follow [Docker Installation](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html) to setup habana runtime for Docker images.

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
2. `cd models/models_v2/pytorch/llama/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    # Default setup to run peak performance with IPEX
    ./setup.sh
    # Choice: If to run peak performance with PyTorch only with inductor (only support greedy search for now)
    source ./inductor/setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)

6. Set INPUT_TOKEN before running the model
   ```
   export INPUT_TOKEN=32
   (choice in [32 64 128 256 512 1024 2016], we prefer to benchmark on 32 and 2016)
   ```

   Set OUTPUT_TOKEN before running the model
   ```
   export OUTPUT_TOKEN=32
   (32 is preferred, while you could set any other length)
   ```
   Set FINETUNED_MODEL to llama2 7b or llama2 13b before running
   ```
   #Test llama2 7b
   export FINETUNED_MODEL="meta-llama/Llama-2-7b-hf"
   #Test llama2 13b
   export FINETUNED_MODEL="meta-llama/Llama-2-13b-hf"
   #Test llama3.1 8b
   export FINETUNED_MODEL="meta-llama/Llama-3.1-8B-Instruct"
   ```
   About the BATCH_SIZE in scripts
   ```
   using BATCH_SIZE=1 for realtime mode
   using BATCH_SIZE=N for throughput mode (N could be further tuned according to the testing host, by default using 1);
   ```
   About the BEAM_SIZE in scripts
   ```
   using BEAM_SIZE=4 by default
   ```
  * Do calibration to get "qconfig.json" before running INT8.
    ```
    #optional: qconfig.json is saved in this repo, you can also do calibration by yourself to re-generation it
    bash do_quantization.sh calibration sq #using smooth quant as default

    #unzip qconfig.zip to get qconfig.json, if you meet error to use this uploaded version of qconfig.zip, please re-generation it as above
    unzip qconfig.zip
    ```
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **FINETUNED_MODEL**    | `#Test llama2 7b: export FINETUNED_MODEL="meta-llama/Llama-2-7b-hf";   #Test llama2 13b: export FINETUNED_MODEL="meta-llama/Llama-2-13b-hf";   # Test llama3.1 8b: export FINETUNED_MODEL="meta-llama/Llama-3.1-8B-Instruct"`         |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, bf16, fp16, int8) |
| **INPUT_TOKEN**    |    `export INPUT_TOKEN=32 (choice in [32 64 128 256 512 1024 2016], we prefer to benchmark on 32 and 2016)`    |
| **OUTPUT_TOKEN**    |   `export OUTPUT_TOKEN=32 (32 is preferred, while you could set any other length)`      |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |


## Output

Single-tile output will typically looks like:

```
2024-05-17 22:35:31,097 - root - INFO - ---------- Summary: ----------
2024-05-17 22:35:31,097 - root - INFO - inference-latency: 18.211 sec.
2024-05-17 22:35:31,097 - root - INFO - first-token-latency: 4.227 sec.
2024-05-17 22:35:31,097 - root - INFO - rest-token-latency: 0.110 sec.
2024-05-17 22:35:31,097 - root - INFO - P90-rest-token-latency: 0.111 sec.
2024-05-17 22:35:36,648 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;total-latency;bf16;1; 18.179000
2024-05-17 22:35:36,655 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;first-token-latency;bf16;1; 4.238500
2024-05-17 22:35:36,664 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;rest-token-latency;bf16;1; 0.110000
2024-05-17 22:35:36,671 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;P90-rest-token-latency;bf16;1; 0.110500
2024-05-17 22:35:36,678 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;token_per_sec;bf16;1; 9.110
2024-05-17 22:35:36,686 - root - INFO - meta-llama/Llama-2-7b-hf;Input/Output Token;1024/128;latency;first_token_thp;bf16;1; 0.236
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
- key: first token throughput
  value: 15.648000
- key: rest token throughput
  value: 0.284250
- key: first token latency
  value: 4.238500
- key: rest_token_latency
  value: 0.110000
- key: accuracy
  value: 93.17
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
