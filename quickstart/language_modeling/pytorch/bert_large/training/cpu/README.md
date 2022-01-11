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

The create_pretraining_data.py script duplicates the input plain text, replaces
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
./parallel_create_hdf5.sh
```
For phase2 the seq_len=512:
```shell
export SEQ_LEN=512
./parallel_create_hdf5.sh
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
python3 ./chop_hdf5_files.py
```

For phase2:

```shell
export SEQ_LEN=512
python3 ./chop_hdf5_files.py
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

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison, Torch-CCL, Jemalloc and TCMalloc.

### Model Specific Setup
* Install dependence
  ```
  pip install datasets accelerate tfrecord
  conda install openblas
  conda install faiss-cpu -c pytorch
  conda install intel-openmp
  ```

* Set ENV to use AMX if you are using SPR
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```
* Set ENV to use multi-nodes distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes. 
  ```
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line 
  ```
## Quick Start Scripts

|  DataType   | Phase 1  |  Phase 2 |
| ----------- | ----------- | ----------- |
| FP32        | bash run_bert_pretrain_phase1.sh fp32 | bash run_bert_pretrain_phase2.sh fp32 |
| BF16        | bash run_bert_pretrain_phase1.sh bf16 | bash run_bert_pretrain_phase2.sh bf16 |

|  DataType   | Distributed Training Phase 1  |  Distributed Training Phase 2 |
| ----------- | ----------- | ----------- |
| FP32        | bash run_ddp_bert_pretrain_phase1.sh fp32 | bash run_ddp_bert_pretrain_phase2.sh fp32 |
| BF16        | bash run_ddp_bert_pretrain_phase1.sh bf16 | bash run_ddp_bert_pretrain_phase2.sh bf16 |
## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory
and an output directory.

```
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Clone the Transformers repo in the BERT large training directory
cd quickstart/language_modeling/pytorch/bert_large/training/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.11.3
git apply ../enable_optmization.diff
pip install -e ./
cd ..

# Env vars
export OUTPUT_DIR=<path to an output directory>
export DATASET_DIR=</path/to/dataset/tfrecord_dir>
export TRAIN_SCRIPT=${MODEL_DIR}/models/language_modeling/pytorch/bert_large/training/run_pretrain_mlperf.py

# For phase 1 get the bert config from https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT
export BERT_MODEL_CONFIG=/path/to/bert_config.json

# Run the phase 1 quickstart script for fp32 (or bf16)
bash run_bert_pretrain_phase1.sh fp32

# For phase 2 set the pretrained model path to the checkpoints generated during phase 1
export PRETRAINED_MODEL=/path/to/bert_large_mlperf_checkpoint/checkpoint/

# Run the phase 2 quickstart script for fp32 (or bf16)
bash run_bert_pretrain_phase2.sh fp32
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

