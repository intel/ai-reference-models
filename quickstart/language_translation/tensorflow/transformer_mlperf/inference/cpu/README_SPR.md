<!--- 0. Title -->
# TensorFlow Transformer Language inference

<!-- 10. Description -->
## Description

This document has instructions for running Transformer Language inference using
Intel-optimized TensorFlow.

## Model Package

The model package includes the Dockerfile and scripts needed to build and
run Transformer Language inference in a container.
```
tf-spr-transformer-mlperf-inference
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-transformer-mlperf-inference.tar.gz
├──tf-spr-transformer-mlperf-inference.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32 or bfloat16). |
| `inference_throughput.sh` | Runs multi instance batch inference (batch-size=64) using 1 instance per socket for the specified precision (fp32 or bfloat16). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32 or bfloat16). |

<!--- 30. Datasets -->
## Datasets

Decide the problem you want to run to get the appropriate dataset.
We will get the training data of it as an example the computing BLEU score:
```
export DATASET_DIR=/home/<user>/transformer_data
mkdir $DATASET_DIR && cd $DATASET_DIR
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
```
Download training/evaluation datasets using [data_download.py script](https://github.com/mlcommons/training/blob/master/translation/tensorflow/transformer/data_download.py) 
```
python data_download.py --data_dir=$DATASET_DIR
```
Once the script completes, you should have a dataset directory `DATASET_DIR` with
the following files: `newstest2014.de`, `newstest2014.en`, and
a vocab text file `vocab.ende.32768`.

More detailed instructions about the model and preparing the dataset can be found [here]((https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)).


## Build the container

The Transformer Language inference package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the Transformer Language inference container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-transformer-mlperf-inference.tar.gz
cd tf-spr-transformer-mlperf-inference

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-transformer-mlperf-inference` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run Transformer Language inference in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. 
By default, the `run.sh` script will run the
`inference_realtime.sh` quickstart script. To run a different script, specify
the name of the script using the `SCRIPT` environment variable.
```
# Navigate to the container package directory
cd tf-spr-transformer-mlperf-inference

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# To run a different script, specify the name of the script using the `SCRIPT` environment variable
SCRIPT=accuracy.sh ./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

