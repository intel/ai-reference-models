<!--- 0. Title -->
# PyTorch T5 inference

<!-- 10. Description -->
## Description

This document has instructions for running [T5](https://huggingface.co/docs/transformers/model_doc/t5) inference using
Intel-optimized PyTorch.

## Bare Metal

### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, and Jemalloc.

### Model Specific Setup

* Install dependency

```bash
pip install transformers
pip install -r requirements.txt
```

* Setup the output dir to store the log

```bash
    export OUTPUT_DIR=$Where_to_save_log
```

* Setup runnning precision

```bash
    export PRECISION=$Data_type(fp32, int8)
```

* Setup model name running

```bash
    export MODEL_NAME=$Model_name(t5-small, t5-base, t5-large, t5-3b and t5-11b)
```

* Setup predict samples to do inference

```bash
    export MAX_PREDICT_SAMPLES=$Max_predict_samples
```

* Setup cores number to use

```bash
    export CORES_PER_INSTANCE=$Cores_use
```

* Set Jemalloc Preload for better performance

The jemalloc should be built from the [General setup](#general-setup) section.

```bash
    export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```

* Set IOMP preload for better performance

IOMP should be installed in your conda env from the [General setup](#general-setup) section.

```bash
    export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use AMX if you are using SPR

```bash
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

## Quick Start Scripts

|  backend   | performance  |
| ----------- | ----------- |
| IPEX        | bash run_inference.sh ipex |
| Offical Pytorch        | bash run_inference.sh pytorch | 
## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variable set to point to an output directory.

```bash
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Env vars
export OUTPUT_DIR=<path to an output directory>
export PRECISION=int8
export MODEL_NAME=t5-small
export MAX_PREDICT_SAMPLES=30
export CORES_PER_INSTANCE=4
# Run a quickstart script (for example, ipex inference)
bash run_inference.sh ipex
```

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
