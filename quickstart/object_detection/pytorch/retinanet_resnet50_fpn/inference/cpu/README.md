# RetinaNet Resnet50 FPN Inference
## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

### Model Specific Setup
* Install dependence
```
    pip install Pillow pycocotools
```

* Download dataset
```
    export DATASET_DIR=#Where_to_save_Dataset
    bash download_dataset.sh
```

* Setup the Output dir to store the log
```
    export OUTPUT_DIR=$Where_to_save_log
```

* Set Jemalloc Preload for better performance

The jemalloc should be built from the [General setup](#general-setup) section.
```
    export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```

* Set IOMP preload for better performance

IOMP should be installed in your conda env from the [General setup](#general-setup) section.
```
    export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use AMX if you are using SPR
```
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

### Inference CMD

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash batch_inference_baremetal.sh fp32 | bash online_inference_baremetal.sh fp32 | bash accuracy_baremetal.sh fp32 |
| BF16        | bash batch_inference_baremetal.sh bf16 | bash online_inference_baremetal.sh bf16 | bash accuracy_baremetal.sh bf16 |

<!--- 80. License -->
## License

[LICENSE](/LICENSE)