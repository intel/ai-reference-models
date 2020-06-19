# ResNet50

The following examples are available for ResNet50 using a model package:
* [FP32 Inference](#fp32-inference)

Note that the ImageNet dataset is used in the ResNet50 examples. To download and preprocess
the ImageNet dataset, see the [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
from the TensorFlow models repo.

## FP32 Inference

### Example Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](/examples/tensorflow/resnet50/fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](/examples/tensorflow/resnet50/fp32_batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`fp32_accuracy.sh`](/examples/tensorflow/resnet50/fp32_accuracy.sh) | Measures the model accuracy (batch_size=100). |

These examples can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

#### Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==2.1.0](https://pypi.org/project/intel-tensorflow/)
* numactl

Download and untar the model package and then run an [example script](#examples). 

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50-fp32-inference.tar.gz
tar -xzf resnet50_fp32_inference.tar.gz
cd resnet50_fp32_inference

examples/<script name>.sh
```

#### Docker

The model container `model-zoo:2.1.0-resnet50-fp32-inference` includes the scripts 
and libraries needed to run ResNet50 v1.5 FP32 training. To run one of the model
training examples using this container, you'll need to provide volume mounts for
the ImageNet dataset and an output directory where checkpoint files will be written.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
        --env DATASET_DIR=${DATASET_DIR} \
        --env OUTPUT_DIR=${OUTPUT_DIR} \
        --env http_proxy=${http_proxy} \
        --env https_proxy=${https_proxy} \
        --volume ${DATASET_DIR}:${DATASET_DIR} \
        --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
        --privileged --init -t \
        amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-resnet50-fp32-inference \
        /bin/bash examples/<script name>.sh
```

### Advanced Options

#### Custom inference runs

To do a custom model training run instead of using a predefined example, the 
[launch_benchmark.py](/docs/general/tensorflow/LaunchBenchmark.md) script can be called
directly. The snippets below demonstrate how to do this in different environments:

*Bare Metal*

Follow the [Bare Metal](#bare-metal) setup above to setup your environment and download
and untar the model package on your machine. Then, call the `launch_benchmarks.py` script
in the `benchmarks` folder:

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50-fp32-inference.tar.gz
tar -xzf resnet50_fp32_inference.tar.gz
cd resnet50_fp32_inference/benchmarks

MODEL_FILE=resnet50_fp32_inference/resnet50_fp32_pretrained_model.pb

python launch_benchmark.py \
     --model-name resnet50 \
     --precision fp32 \
     --mode inference \
     --batch-size=128 \
     --socket-id 0 \
     --framework tensorflow \
     --output-dir {OUTPUT_DIR} \
     --in-graph ${MODEL_FILE} \
     --data-location ${DATASET_DIR}
```

*Docker*

Similarly to running the [docker example](#docker) above, the container needs to be run with
volume mounts for the dataset and an output directory where log files will be
written. Instead of having the container run one of the example scripts, the `launch_benchmark.py`
script in the `benchmarks` folder can be called directly.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
    --volume ${DATASET_DIR}:${DATASET_DIR} \
    --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    --privileged --init -t \
    amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-resnet50-fp32-inference \
    python benchmarks/launch_benchmark.py \
        --model-name resnet50 \
        --precision fp32 \
        --mode inference \
        --batch-size=128 \
        --socket-id 0 \
        --framework tensorflow \
        --output-dir {OUTPUT_DIR} \
        --in-graph resnet50_fp32_pretrained_model.pb \
        --data-location ${DATASET_DIR}
```

#### Using a local copy of the model package with docker

A download of the model package can be run in a docker container by mounting the
model package directory to `/workspace` container and using the image recognition
category container.

<pre>
DATASET_DIR=&lt;path to the preprocessed imagenet dataset&gt;
OUTPUT_DIR=&lt;directory where log files will be written&gt;

wget https://ubit-artifactory-or.intel.com/artifactory/list/cicd-or-local/model-zoo/resnet50-fp32-inference.tar.gz
tar -xzf resnet50_fp32_inference.tar.gz

docker run \
    --env DATASET_DIR=${DATASET_DIR} \
    --env OUTPUT_DIR=${OUTPUT_DIR} \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    <mark><b>--volume resnet50_fp32_inference:/workspace</b></mark> \
    --volume ${DATASET_DIR}:${DATASET_DIR} \
    --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
    --privileged --init -w /workspace -t \
    <mark><b>amr-registry.caas.intel.com/aipg-tf/model-zoo:2.1.0-resnet50-fp32-inference</b></mark> \
    /bin/bash /workspace/examples/&lt;script name&gt;.sh
</pre>


