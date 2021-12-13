<!--- 0. Title -->
# PyTorch BERT Large inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large SQuAD1.1 inference using
Intel-optimized PyTorch.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run BERT Large inference in a container.
```
pytorch-spr-bert-large-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── pytorch-spr-bert-large-inference.tar.gz
├── pytorch-spr-bert-large-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `run_multi_instance_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |
| `run_multi_instance_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |
| `run_accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, avx-fp32, int8, avx-int8, or bf16) using the [huggingface fine tuned model](https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin). |

> Note: The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.

## Datasets

Please following this [link](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering)
to get `dev-v1.1.json` and set the `EVAL_DATA_FILE` environment variable to point
to the file:
```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
export EVAL_DATA_FILE=$(pwd)/dev-v1.1.json
```

## Build the container

The BERT Large inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the PyTorch/IPEX container as it's base, so ensure that you have built
the `pytorch-ipex-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep pytorch-ipex-spr
model-zoo         pytorch-ipex-spr         fecc7096a11e        40 minutes ago      8.31GB
```

To build the BERT Large inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf pytorch-spr-bert-large-inference.tar.gz
cd pytorch-spr-bert-large-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:pytorch-bert-large-inference` that will be used to run the model.

## Run the model

Download the `config.json` and fine tuned model from huggingface and set the `PRETRAINED_MODEL`
environment variable to point to the directory that has both files:
```
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
PRETRAINED_MODEL=$(pwd)/bert_squad_model
```

Once you have the pretarined model and have [built the container](#build-the-container),
use the `run.sh` script from the container package to run BERT Large inference in docker.
Set environment variables to specify the precision to run, and an output directory.
By default, the `run.sh` script will run the `run_multi_instance_realtime.sh` quickstart script.
To run a different script, specify the name of the script using the `SCRIPT` environment
variable.
```
# Navigate to the container package directory
cd pytorch-spr-bert-large-inference

# Set the required environment vars
export PRETRAINED_MODEL=<path to the downloaded model files>
export EVAL_DATA_FILE=<path to the dev-v1.1.json file>
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Use the SCRIPT env var to run a different quickstart script
SCRIPT=run_accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

