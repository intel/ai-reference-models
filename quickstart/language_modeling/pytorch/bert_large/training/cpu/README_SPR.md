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

BERT Large training uses the config file and enwiki-20200101 dataset from the
[MLCommons training GitHub repo](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert).

Follow the instructions in their documentation to download the files and
preprocess the dataset to create TF records files. Set the `DATASET_DIR`
environment variable to the path to the TF records directory. Your directory
should look similar like this:
```
<DATASET_DIR>
├── seq_128
│   └── part-00000-of-00500_128
└── seq_512
    └── part-00000-of-00500
```

Download the `bert_config.json` file from the Google drive that is linked at the
[MLCommons BERT README](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#location-of-the-input-files).
Set the `CONFIG_FILE` environment variable the path to the downloaded file
when running the phase 1 quickstart script.
```
export CONFIG_FILE=<path to bert_config.json>
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
To run phase 2, use the model_save from the phase 1 output as the `CHECKPOINT_DIR`.
Alternatively, checkpoints can be downloaded from online (for example, from the
[MLCommons repo](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert),
but these are TF checkpoints that would need to be converted to a PyTorch model file).
```
# To run phase 2, set the CHECKPOINT_DIR to the model_save directory from phase 1's output
export CHECKPOINT_DIR=${OUTPUT_DIR}/model_save

# Define a new directory for phase 2 output, and set the SCRIPT var to run phase 2
export OUTPUT_DIR=<directory where log and model files will be written for phase 2>
SCRIPT=run_bert_pretrain_phase2.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

