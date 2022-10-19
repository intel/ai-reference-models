<!--- 0. Title -->
# PyTorch T5 inference

<!-- 10. Description -->
## Description

This document has instructions for running [T5](https://huggingface.co/docs/transformers/model_doc/t5) inference using
Intel-optimized PyTorch.

Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you install dependencies and have a clone of the [Model Zoo Github repository](https://github.com/IntelAI/models).

* Install dependencies

```bash
pip install transformers
pip install -r requirements.txt
```
* Clone Model Zoo repo
```
git clone https://github.com/IntelAI/models.git
```

## Run on Linux

### Quick Start Scripts

|  backend   | performance  |
| ----------- | ----------- |
| IPEX        | bash run_inference.sh ipex |
| Offical Pytorch        | bash run_inference.sh pytorch | 

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

* Set Jemalloc Preload for better performance

  After [Jemalloc setup](/docs/general/pytorch/BareMetalSetup.md#build-jemalloc), set the following environment variables.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env. Set the following environment variables.
  ```
  export LD_PRELOAD=<path to the intel-openmp directory>/lib/libiomp5.so:$LD_PRELOAD
  ```

* Set ENV to use AMX if you are using SPR
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

* Run the model:
  ```
  cd models

    # Set environment variables
    export OUTPUT_DIR=<path to an output directory>
    export PRECISION=fp32(for example, fp32, int8)
    export MODEL_NAME=t5-small(for example, t5-small, t5-base, t5-large, t5-3b and t5-11b)
    export MAX_PREDICT_SAMPLES=30(Setup predict samples to do inference)
    export CORES_PER_INSTANCE=4(Setup cores number to use)
    
    # Run a quickstart script (for example, ipex inference)
    bash quickstart/language_modeling/pytorch/t5/inference/cpu/run_inference.sh ipex
  ```

## Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Using Windows CMD.exe, run:
```
cd models

#Set environment variables
set OUTPUT_DIR=<path to an output directory>
set PRECISION=fp32(for example, fp32, int8)
set MODEL_NAME=t5-small(for example, t5-small, t5-base, t5-large, t5-3b and t5-11b)
set MAX_PREDICT_SAMPLES=30(Setup predict samples to do inference)
set CORES_PER_INSTANCE=4(Setup cores number to use)

#Run a quickstart script (FP32 online inference or batch inference or accuracy)
bash quickstart\language_modeling\pytorch\t5\inference\cpu\run_inference.sh 

```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
