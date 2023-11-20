<!--- 0. Title -->
# PyTorch BERT Large training
<!-- 10. Description -->
## Description

This document has instructions for running BERT Large pre-training using
Intel-optimized PyTorch.

## Datasets

# Location of the input files 

This [MLCommons members Google Drive location](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) contains the following.
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

# Checkpoint conversion
python convert_tf_checkpoint.py --tf_checkpoint /cks/model.ckpt-28252 --bert_config_path /cks/bert_config.json --output_checkpoint model.ckpt-28252.pt

# Download the preprocessed text dataset

From the [MLCommons BERT Processed dataset
directory](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v?usp=sharing)
download `results_text.tar.gz`, and `bert_reference_results_text_md5.txt`.  Then perform the following steps:

```shell
tar xf results_text.tar.gz
cd results4
md5sum --check ../bert_reference_results_text_md5.txt
cd ..
```

After completing this step you should have a directory called `results4/` that
contains 502 files for a total of about 13Gbytes.

# Generate the BERT input dataset

The [create_pretraining_data.py](/models/language_modeling/pytorch/bert_large/training/input_preprocessing/create_pretraining_data.py) script duplicates the input plain text, replaces
different sets of words with masks for each duplication, and serializes the
output into the HDF5 file format.

## Training data

The following shows how create_pretraining_data.py is called by a parallelized
script that can be called as shown below.  The script reads the text data from
the `results4/` subdirectory and outputs the resulting 500 hdf5 files to a
subdirectory named `hdf5/`.

```shell
pip install tensorflow-cpu
```

For phase1 the seq_len=128:
```shell
export SEQ_LEN=128
./models/language_modeling/pytorch/bert_large/training/input_preprocessing/parallel_create_hdf5.sh
```
For phase2 the seq_len=512:
```shell
export SEQ_LEN=512
./models/language_modeling/pytorch/bert_large/training/input_preprocessing/parallel_create_hdf5.sh
```

The resulting `hdf5/` subdir will have 500 files named
`part-00???-of-0500.hdf5` and have a size of about 539 Gigabytes.

Next we need to shard the data into 2048 chunks.  This is done by calling the
chop_hdf5_files.py script.  This script reads the 500 hdf5 files from
subdirectory `hdf5/` and creates 2048 hdf5 files in subdirectory
`2048_shards_uncompressed`.

For phase1:

```shell
export SEQ_LEN=128
python3 ./models/language_modeling/pytorch/bert_large/training/input_preprocessing/chop_hdf5_files.py
```

For phase2:

```shell
export SEQ_LEN=512
python3 ./models/language_modeling/pytorch/bert_large/training/input_preprocessing/chop_hdf5_files.py
```

The above will produce a subdirectory named `2048_shards_uncompressed/`
containing 2048 files named `part_*_of_2048.hdf5` and have a size of about 539 Gigabytes.
you can use "SHARD_NUM" to control the shard files number. the default "SHARD_NUM" if 2048. 

```
<DATASET_DIR>
├── 2048_shards_uncompressed_512
│   └── part-00000-of-00xxx
└── 2048_shards_uncompressed_128
    └── part-00000-of-00xxx
```

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Model Specific Setup

* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
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

* Set ENV to use multi-nodes distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes. 
  ```
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line 
  ```

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `run_bert_pretrain_phase1.sh` | Runs BERT large pretraining phase 1 using max_seq_len=128 for the first 90% dataset for the specified precision (fp32, avx-fp32, bf32 or bf16). The script saves the model to the `OUTPUT_DIR` in a directory called `model_save`. |
| `run_bert_pretrain_phase2.sh` | Runs BERT large pretraining phase 2 using max_seq_len=512 with the remaining 10% of the dataset for the specified precision (fp32, avx-fp32, bf32 or bf16). Use path to the `model_save` directory from phase one as the `CHECKPOINT_DIR` for phase 2. |

|  DataType   | Distributed Training Phase 1  |  Distributed Training Phase 2 |
| ----------- | ----------- | ----------- |
| FP32        | bash run_ddp_bert_pretrain_phase1.sh fp32 | bash run_ddp_bert_pretrain_phase2.sh fp32 |
| BF32        | bash run_ddp_bert_pretrain_phase1.sh bf32 | bash run_ddp_bert_pretrain_phase2.sh bf32 |
| BF16        | bash run_ddp_bert_pretrain_phase1.sh bf16 | bash run_ddp_bert_pretrain_phase2.sh bf16 |

## Quick Start Script for fast_bert with TPP optimization 
|  DataType   | pre-train  |  finetune|
| ----------- | ----------- | ----------- |
| BF16        |bash sh run_fast_bert_pretrain.sh | bash sh fast_bert_squad_finetune.sh --use_tpp --tpp_bf16 --unpad |

```
bash sh run_fast_bert_pretrain.sh
```
will run 32 ranks on 8 nodes with `global batch size=2048`.
You can config following ENV to run more settings, for example
```
export NPROCESS=16
export PROCESS_PER_NODE=2
export GBS=1024
```
will run 16 ranks on 8 nodes with `global batch size=1024`.

**Note**: The `avx-fp32` precision runs the same scripts as `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory
and an output directory.

```
# Clone the Intel® AI Reference Models repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Install dependencies:
./quickstart/language_modeling/pytorch/bert_large/training/cpu/setup.sh

# Env vars
export OUTPUT_DIR=<path to an output directory>
export DATASET_DIR=<path to the dataset>
export TRAIN_SCRIPT=${MODEL_DIR}/models/language_modeling/pytorch/bert_large/training/run_pretrain_mlperf.py
export PRECISION=<specify the precision to run: fp32, avx-fp32, bf16 or bf32>

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# For phase 1 get the bert_config.json from [here](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)
wget https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT
export BERT_MODEL_CONFIG=$(pwd)/bert_config.json

# Run the phase 1 quickstart script:
# This downloads the checkpoints in 'CHECKPOINT_DIR'
export CHECKPOINT_DIR=$(pwd)/checkpoint_phase1_dir
./quickstart/language_modeling/pytorch/bert_large/training/cpu/run_bert_pretrain_phase1.sh

# For phase 2 set the pretrained model path to the checkpoints generated during phase 1
export PRETRAINED_MODEL=$(pwd)/checkpoint_phase1_dir

# Run the phase 2 quickstart script:
./quickstart/language_modeling/pytorch/bert_large/training/cpu/run_bert_pretrain_phase2.sh
```

## CheckPoint in Your Training Phase 1 or Phase 2

By default it is `--skip_checkpoint` in run_bert_pretrain_phase2.sh.
To configure checkpoint for Phase 1 or Phase 2, you could set the following args properly according to your training samples:

```
--min_samples_to_start_checkpoints // "Number of update steps until model checkpoints start saving to disk."
--num_samples_per_checkpoint  // "Number of update steps until a model checkpoint is saved to disk."
--log_freq  // "frequency of logging loss. If not positive, no logging is provided for training loss"
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

