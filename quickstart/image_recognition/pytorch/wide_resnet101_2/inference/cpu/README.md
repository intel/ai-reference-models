# WIDE\_RESNET101\_2 Inference
## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

## Datasets

### ImageNet

The [ImageNet](http://www.image-net.org/) validation dataset is used to run WIDE\_RESNET101\_2
accuracy tests.

Download and extract the ImageNet2012 dataset from [http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:
```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the
`DATASET_DIR` (for example: `export DATASET_DIR=/home/<user>/imagenet`).

### Model Specific Setup

* Setup the model dir, dataset dir and output dir:
```bash
    export MODEL_DIR=#path_to_frameworks.ai.models.intel-models
    export DATASET_DIR=#path_to_Imagenet_Dataset
    export OUTPUT_DIR=#Where_to_save_log
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

### Inference CMD

#### Use IPEX

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash batch_inference_baremetal.sh fp32 | bash online_inference_baremetal.sh fp32 | bash accuracy_baremetal.sh fp32 |
| BF16        | bash batch_inference_baremetal.sh bf16 | bash online_inference_baremetal.sh bf16 | bash accuracy_baremetal.sh bf16 |

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
