<!--- 0. Title -->
# PyTorch DLRM inference

<!-- 10. Description -->
## Description

This document has instructions for running DLRM inference using
Intel-optimized PyTorch for bare metal.

### General Setup
Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, and Jemalloc.

### Datasets and Pre-trained Weight
Follow the same step in [link](/quickstart/recommendation/pytorch/dlrm/inference/cpu/README_SPR.md) to prepare datasets and weights.

### Model Specific Setup
```sh
pip install ../../requirements.txt
```
### Environment Setup
```sh
# jemalloc
export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
# iomp
export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
# enable AMX
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX # only needed on SPR
# set OUTPUT_DIR, PRECISION, WEIGHT, DATASET
export PRECISION=<specify the precision to run>
export WEIGHT_PATH=<path to the tb00_40M.pt file> # only needed for testing accuracy
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

```
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `bare_metal_performance.sh` | Run inference to verify performance for the specified precision (fp32, int8, avx-int8, or bf16). |
| `bare_metal_accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, int8, avx-int8, or bf16). |

## Running CMD
```sh
bash bare_metal_performance.sh # for testing performance
bash bare_metal_accuracy.sh # for testing accuracy
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.