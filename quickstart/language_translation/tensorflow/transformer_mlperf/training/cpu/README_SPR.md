<!--- 0. Title -->
# TensorFlow Transformer Language training

<!-- 10. Description -->

This document has instructions for running Transformer Language training
using Intel-optimized TensorFlow.


## Model Package

The model package includes the Dockerfile and scripts needed to build and
run Transformer Language training in a container.
```
tf-spr-transformer-mlperf-training
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-transformer-mlperf-training.tar.gz
├──tf-spr-transformer-mlperf-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 5120 for the specified precision (fp32 or bfloat16). Logs for each instance are saved to the output directory. |

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

The Transformer Language training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the Transformer Language training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-transformer-mlperf-training.tar.gz
cd tf-spr-transformer-mlperf-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-transformer-mlperf-training` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run Transformer Language training in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`training.sh` quickstart script.
```
# Navigate to the container package directory
cd tf-spr-transformer-mlperf-training

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with training.sh quickstart script
./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

