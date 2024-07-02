# DistilBERT Inference

DistilBERT Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/huggingface/transformers/tree/main/src/transformers/models/distilbert        |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)
* Installation of [Build PyTorch + IPEX + TorchVision Jemalloc and TCMalloc](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md)
* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc and tcmalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
* Set IOMP preload for better performance
```
  pip install packaging intel-openmp
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

* Set ENV to use fp16 AMX if you are using a supported platform
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

# About the Model:
* Set SEQUENCE_LENGTH before running the model
  ```
  export SEQUENCE_LENGTH=128
  (128 is preferred, while you could set any other length)
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

  Note: If you would have a SPR-56C host, BATCH_SIZE=205 is perferred for INT8-BF16 Throughput mode and BATCH_SIZE=198 is perferred for BF16 Throughput mode.
  Customized BATCH_SIZE is supposed to be no larger than dataset size 872.
  ```

* Do calibration to get quantization config before running INT8 (Default attached is produced with sequence length 128).
  ```
  #Set the SEQUENCE_LENGTH to which is going to run when doing the calibration.
  bash do_calibration.sh
  ```
* [Optional for offline tests] Prepare model and dataset files locally
  ```
  (1) download model and sst2 dataset (make sure to install git-lfs first by apt-get install git-lfs)
  bash download_model_dataset.sh
  #by default they are downloaded in current path
  #note that you should do this after you prepared model (transformers repo)

  (2) make following changes in the scirpts to run:
  delete: --task_name sst2  ==>  add: --train_file {path/to/data_file}/SST-2/train.csv --validation_file {path/to/data_file}/SST-2/dev.csv

  (3) export model path
  export FINETUNED_MODEL={path/to/model_file}/distilbert-base-uncased-finetuned-sst-2-english

  (4) run scirpt with HF_DATASETS_OFFLINE=1 flag, like:
  HF_DATASETS_OFFLINE=1 bash run_multi_instance_throughput.sh fp32

  ```

# Prepare Dataset
Use the following instructions to download the SST-2 dataset.
Also, clone the Intel® AI Reference Models GitHub Repository:
```
git clone https://github.com/IntelAI/models.git
cd models
export DATASET_DIR=<path to download the dataset>
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip -O $DATASET_DIR/SST-2.zip
unzip $DATASET_DIR/SST-2.zip -d $DATASET_DIR/
python models_v2/distilbert_base/inference/cpu/convert.py $DATASET_DIR
```

# Pre-Trained Model
Follow the instructions below to download the pre-trained model.
```
git clone https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
export FINETUNED_MODEL=$(pwd)/distilbert-base-uncased-finetuned-sst-2-english
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/distilbert/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

5. Run setup scripts:
  ```
  ./setup.sh
  ```

6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE**              |                     `export TEST_MODE=THROUGHPUT <select from: THROUGHPUT< REALTIME, ACCURACY>`                                  |
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-sst-2>`                                  |
| **PRECISION**    |                               `export PRECISION=fp32 <select from :- fp32, bf32, bf16, int8-fp32, int8-bf16>`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=$PWD`                               |
| **FINETUNED_MODEL**    |                               `export FINETUNED_MODEL=<path to pre-trained model>`                               |
| **SEQUENCE_LENGTH**    |                               `export SEQUENCE_LENGTH=128`                        |
| **HF_DATASETS_OFFLINE**    |                               `export HF_DATASETS_OFFLINE=0`                               |
| **CORE_PER_INSTANCE**   |          `export CORE_PER_INSTANCE=4` |
| **BATCH_SIZE** (optional)  |                               `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |

6. Run `run_model.sh`
## Output


```
***** eval metrics *****
  eval_accuracy             =     0.9106
  eval_jit_compilation_time =     0.6645
  eval_loss                 =     0.3902
  eval_runtime              = 0:00:08.27
  eval_samples              =        872
  eval_samples_per_second   =    105.353
  eval_steps_per_second     =    105.353
```


Final results of the inference run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 105.353
  unit: sentences per second
- key: latency
  value: 9.5238
  unit: ms
- key: accuracy
  value: 0.9106
  unit: percentage
```
