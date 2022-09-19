<!--- 0. Title -->
# PyTorch BERT Large training

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large pre-training using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run BERT Large training in a container.
```
pytorch-spr-bert-large-training
├── README.md
├── build.sh
├── input_preprocessing
│   ├── chop_hdf5_files.py
│   ├── create_pretraining_data.py
│   ├── create_pretraining_data_wrapper.sh
│   ├── prallel_create_hdf5.sh
│   └── tokenization.py
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-spr-bert-large-training.tar.gz
├── pytorch-spr-bert-large-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_bert_pretrain_phase1.sh` | Runs BERT large pretraining phase 1 using max_seq_len=128 for the first 90% dataset for the specified precision (fp32, avx-fp32, or bf16). The script saves the model to the `OUTPUT_DIR` in a directory called `model_save`. |
| `run_bert_pretrain_phase2.sh` | Runs BERT large pretraining phase 2 using max_seq_len=512 with the remaining 10% of the dataset for the specified precision (fp32, avx-fp32, or bf16). Use path to the `model_save` directory from phase one as the `CHECKPOINT_DIR` for phase 2. |

> Note: The `avx-fp32` precision runs the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets

### Input files

Download the following files from the [MLCommons members Google Drive](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT):
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

### Download and extract the dataset

From the [MLCommons BERT Processed dataset
directory](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v?usp=sharing)
download the `results_text.tar.gz` and `bert_reference_results_text_md5.txt` files. Perform the
following steps to extract the tar file and check the MD5 checksums.
```shell
tar xf results_text.tar.gz
cd results4
md5sum --check ../bert_reference_results_text_md5.txt
cd ..
```
After completing this step you should have a directory called `results4/` that
contains 502 files for a total of about 13Gbytes.

### Generate the BERT input dataset

There are input preprocessing scripts in the `pytorch-spr-bert-large-training/input_preprocessing` folder.
Note that the pretraining dataset is about `539GB`, so ensure that you have enough disk
space. First, navigate to the folder with the preprocessing scripts.
```
cd pytorch-spr-bert-large-training/input_preprocessing
```
> Note that the `results4` folder and the `vocab.txt` file will need to be in the
> `input_preprocessing` before running the preprocessing scripts.

The `create_pretraining_data.py` script duplicates the input plain text, replaces
different sets of words with masks for each duplication, and serializes the
output into the HDF5 file format.

The following snippet shows how `create_pretraining_data.py` is called by a parallelized
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

The resulting `hdf5` subdirs will have 500 files named
`part-00???-of-0500.hdf5` and have a size of about 539 Gigabytes.

Next we need to shard the data into 2048 chunks.  This is done by calling the
chop_hdf5_files.py script.  This script reads the 500 hdf5 files from
subdirectory `hdf5` and creates 2048 hdf5 files in subdirectory
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

## Build the container

The BERT Large training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         fecc7096a11e        40 minutes ago      8.31GB
```

To build the BERT Large training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-bert-large-training.tar.gz
cd pytorch-spr-bert-large-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:pytorch-bert-large-training` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container),
download the [dataset and config file](#datasets), use the `run.sh` script from the
container package to run BERT Large training in docker. Set environment variables to
specify the precision to run, dataset directory, config file directory, and an
output directory. Use an empty `OUTPUT_DIR` to start with to prevent any previously
generated checkpoints from getting picked up. By default, the `run.sh` script will
run the `run_bert_pretrain_phase1.sh` quickstart script.
```
# Navigate to the container package directory
cd pytorch-spr-bert-large-training

# Set the required environment vars to run phase 1
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log and model files will be written for phase 1>
export DATASET_DIR=<path to the preprocessed dataset>
export CONFIG_FILE=<path to the bert_config.json>

# Run the container with the default run_bert_pretrain_phase1.sh quickstart script
./run.sh
```
To run phase 2, to the directory where checkpoints were generated during phase 1
pretraining. This `CHECKPOINT_DIR` should also have the `bert_config.json` file.
```
# To run phase 2, set the CHECKPOINT_DIR to the folder with checkpoints generated during phase 1
export CHECKPOINT_DIR=<directory with checkpoints and the bert_config.json file>

# Define a new directory for phase 2 output, and set the SCRIPT var to run phase 2
export OUTPUT_DIR=<directory where log and model files will be written for phase 2>
SCRIPT=run_bert_pretrain_phase2.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

