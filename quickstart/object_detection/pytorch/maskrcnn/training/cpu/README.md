# Mask-RCNN Training
## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

### Model Specific Setup

* Install dependencies
```
    pip install yacs opencv-python pycocotools cityscapesscripts
    conda install intel-openmp
```

* Install model
```
    cd models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/
    python setup.py develop
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

### Training CMD

|  DataType   | Throughput  |
| ----------- | ----------- |
| FP32        | bash train_throughput_wip.sh fp32 |
| BF16        | bash train_throughput_wip.sh bf16 |

## Docker

Make sure, you have all the requirements pre-setup in your Container as the [Bare Metal](#bare-metal) Setup section.

### Download dataset, backbone weight and pretrained model

Refer to the corresponding Bare Mental Section to download the dataset, weight and models

### Running CMD
```
DATASET_DIR=$dataset_dir
OUTPUT_DIR=$Where_to_save_the_log

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/object-detection:pytorch-latest-maskrcnn-training \
  /bin/bash quickstart/<script name>.sh <data_type>
```
If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)