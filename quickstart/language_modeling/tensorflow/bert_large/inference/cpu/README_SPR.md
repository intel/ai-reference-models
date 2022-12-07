<!--- 0. Title -->
# TensorFlow BERT Large inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large inference using
Intel-optimized TensorFlow.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run BERT Large inference in a container.
```
tf-spr-bert-large-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-bert-large-inference.tar.gz
├──tf-spr-bert-large-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32, int8, bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_realtime_weight_sharing.sh` | Runs multi instance realtime inference with weight sharing for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32, int8, bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput.sh` | Runs multi instance batch inference for BERT large (SQuAD) using 1 instance per socket with batch size 128 (for precisions: fp32, int8 or bfloat16). Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures BERT large (SQuAD) inference accuracy for the specified precision (fp32, int8 or bfloat16) with batch size 56. |

<!--- 30. Datasets -->
## Datasets

### BERT Large Data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
Set the `DATASET_DIR` to point to that directory when running BERT Large inference using the SQuAD data.

## Build the container

The BERT Large inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the BERT Large inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-bert-large-inference.tar.gz
cd tf-spr-bert-large-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-bert-large-inference` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run BERT Large inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd tf-spr-bert-large-inference

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the downloaded pre-trained model>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Use the SCRIPT env var to run a different quickstart script
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

