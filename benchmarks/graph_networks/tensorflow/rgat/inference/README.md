<!--- 0. Title -->
# RGAT Inference

<!-- 10. Description -->

## Description
This document has instructions for running inference using the Relational Graph Attention Networks (RGAT) model. The model is based on the paper titled [Relational Graph Attention Networks](https://arxiv.org/abs/1904.05811). Relational Graph Attention Networks are used for node classification tasks in relational data represented by graphs.

## Dataset

OGBN-MAG is [Open Graph Benchmark](https://ogb.stanford.edu/)'s node classification task on a subset of the [Microsoft Academic Graph](https://www.microsoft.com/en-us/research/publication/microsoft-academic-graph-when-experts-are-not-enough/).

1. ### Download and extract the graph and JSON sampling schemas
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_11_0/RGAT-schemas.tar.gz
tar -xf RGAT-schemas.tar.gz
```

2. ### Download and extract the pretrained model
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/2_11_0/RGAT-pretrained-model.tar.gz
tar -xf RGAT-pretrained-model.tar.gz
```

3. ### Download and preprocess the dataset

i. Create a virtual environment, activate and install the necessary packages.
```
pip install six wheel mock numpy packaging requests pyyaml
pip install keras_preprocessing --no-deps
pip install tensorflow_gnn scipy ogb
```
ii. NOTE: the step below is broken and under investigation. Please contact TF dev team.
Run the ['download_and_preprocess_data.py`](../../../../../models/graph_networks/tensorflow/rgat/download_and_preprocess_data.py) using the following command:

```
python download_and_preprocess_data.py --dataset ogbn_mag --graph_schema_dir <directory-containing-graph-schemas-(.json, .pbtxt)> --ogb_dir <directory-location-to-save-the-final-dataset>
```

## Run the model

### Run on Linux

Install the Intel-optimized TensorFlow along with model dependencies under [requirements.txt](../../../../../models/graph_networks/tensorflow/rgat/inference/requirements.txt). NOTE that the scripts assume TensorFlow is already installed in the runtime environment.

```
# cd to your model zoo directory
cd models
export GRAPH_SCHEMA_PATH=<path to the ogbn_mag_subgraph_schema.pbtxt downloaded and extracted in step 1>
export PRETRAINED_MODEL=<path to the pretrained model directory downloaded and extracted in step 2>
export DATASET_DIR=<path to the test data downloaded and preprocessed in step 3>
export PRECISION=<set the precision to "fp32", "bfloat16", "fp16" or "bfloat32">
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>
```

### Inference
1. `inference.sh`
Runs realtime inference using a default `batch_size=1` for the specified precision (fp32, bfloat16, fp16 or bfloat32). To run inference for throughtput, set `BATCH_SIZE` environment variable.
```
./quickstart/graph_networks/tensorflow/rgat/inference/cpu/inference.sh
```

2. `inference_realtime_multi_instance.sh`
Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, bfloat16, fp16 or bfloat32) with 200 steps. Waits for all instances to complete, then prints a summarized throughput value.
```
./quickstart/graph_networks/tensorflow/rgat/inference/cpu/inference_realtime_multi_instance.sh
```

3. `inference_throughput_multi_instance.sh`
Runs multi instance batch inference using 1 socket per instance for the specified precision (fp32, bfloat16, fp16 or bfloat32) with 200 steps. Waits for all instances to complete, then prints a summarized throughput value.
```
./quickstart/graph_networks/tensorflow/rgat/inference/cpu/inference_throughput_multi_instance.sh
```

### Accuracy
```
./quickstart/graph_networks/tensorflow/rgat/inference/cpu/accuracy.sh
```

### Appendix
1. If you run into bazel not found issues - one way to install is
```
pip install nodeenv
nodeenv -p
npm install -g @bazel/bazelisk
```
