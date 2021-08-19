 # LPOT Containers with Intel® Optimizations for TensorFlow

The dockerfiles in this directory use the
[intel/intel-optimized-tensorflow](https://hub.docker.com/r/intel/intel-optimized-tensorflow)
images as their base, and include an install of the
[Intel® Low Precision Optimization Tool](https://github.com/intel/lpot).
The model-specific dockerfiles also include the pretrained model to allow running the
[LPOT TensorFlow examples](https://github.com/intel/lpot/tree/master/examples/tensorflow)
to demonstrate how the tool quantizes the frozen graph.

Docker pull commands:
```
docker pull amr-registry.caas.intel.com/aipg-tf/model-zoo-ci:134-ci-build-2.3.0-ubuntu-20.04-tf-lpot

docker pull amr-registry.caas.intel.com/aipg-tf/model-zoo-ci:134-ci-build-2.3.0-ubuntu-20.04-tf-lpot-inceptionv3
```

## Building the containers

If you would like to build your own LPOT container, this section has instructions
on how to do that. The docker containers can be built using either the dockerfiles
in this directory or the [model-builder](/tools/scripts/model-builder):

* Build the LPOT container from the dockerfile using the following command:
 ```
 docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$http_proxy \
    --build-arg TENSORFLOW_TAG=2.3.0-ubuntu-20.04 \
    --build-arg PY_VERSION=3.7 \
    -f intel-tf-lpot.Dockerfile \
    -t intel-optimized-tensorflow:2.3.0-ubuntu-20.04-lpot .
 ```
 To build the model-specific dockerfiles, substitute in the name the dockerfile
 that you want to build, and update the name in the `-t` arg to the name your container.

* To build the LPOT containers using the model-builder, first follow the
 [instructions for getting your environment setup to run the script](https://gitlab.devtools.intel.com/intelai/models/-/tree/develop/tools#model-builder-setup).
 After the setup is done, you can build all the LPOT containers using:
 ```
 model-builder --verbose build -f lpot
 ```
 To build a single container, specify the name of the spec like:
 ```
 model-builder --verbose build -f lpot <spec name>
 ```

## Running the container

### Running the TensorFlow LPOT Container

This container has Intel-optimized TensorFlow and LPOT installed, and it
includes a clone of the [LPOT repo](https://github.com/intel/lpot/) at `/src/lpot`.
There are [examples](https://github.com/intel/lpot/tree/master/examples/tensorflow)
that you can run in the LPOT repo, or you can use this container to run
quantization on your own model.

For example, the command snippet below shows how to use this container to run the
image recognition tuning script on a ResNet v1.5 frozen graph and config yaml file
mounted from a directory on the system. The directory that has the model and config
file are being mounted in the container as the `MODEL_DIR`. In addition to the model
directory, the dataset directory and an output folder are also being mounted in the
container. The tuning script is being run in the container with the parameters
pointing to the model and config file in the `MODEL_DIR`.
```
MODEL_DIR=<folder with the frozen graph and config yaml>
DATASET_DIR=<dataset path being referenced in the config yaml>
OUTPUT_DIR=<directory for output files>

docker run \
  --env http_proxy=$http_proxy \
  --env https_proxy=$https_proxy \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MODEL_DIR=${MODEL_DIR} \
  -v ${DATASET_DIR}:${DATASET_DIR} \
  -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
  -v ${MODEL_DIR}:${MODEL_DIR} \
  -w /src/lpot/examples/tensorflow/image_recognition \
  -it amr-registry.caas.intel.com/aipg-tf/model-zoo-ci:134-ci-build-2.3.0-ubuntu-20.04-tf-lpot \
  /bin/bash run_tuning.sh --config=${MODEL_DIR}/resnet50_v1_5.yaml \
  --input_model=${MODEL_DIR}/resnet50_v1.pb \
  --output_model=${OUTPUT_DIR}/lpot_resnet50_v15.pb
```

For more information on running LPOT, see the instructions and getting
started links in the [LPOT repo](https://github.com/intel/lpot#getting-started).

### TensorFlow InceptionV3 LPOT Container

This container has Intel-optimized TensorFlow and LPOT installed, a clone of
the [LPOT repo](https://github.com/intel/lpot/) at `/src/lpot`, and
it includes the InceptionV3 FP32 pretrained model frozen graph file.
The config file for InceptionV3 references a dataset to use for calibration and
evaluation. In this example, the [ImageNet dataset](http://www.image-net.org)
is used. For information on downloading and preparing the dataset, see
the [instructions here](/datasets/imagenet/README.md).

Run the container interactively:
```
DATASET_DIR=<path to the ImageNet dataset directory>
OUTPUT_DIR=<directory for output files>

docker run \
  --env http_proxy=$http_proxy \
  --env https_proxy=$https_proxy \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  -v ${DATASET_DIR}:${DATASET_DIR} \
  -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
  -it amr-registry.caas.intel.com/aipg-tf/model-zoo-ci:134-ci-build-2.3.0-ubuntu-20.04-tf-lpot-inceptionv3 \
  /bin/bash
```
When the container is launched the working directory is the
[lpot/examples/tensorflow/image_recognition](https://github.com/intel/lpot/tree/master/examples/tensorflow/image_recognition)
folder in a clone of the [LPOT repo](https://github.com/intel/lpot/). The
pretrained model called `inceptionv3_fp32_pretrained_model.pb` has also
been added to this folder.

If you don't have your own config file, edit the `inception_v3.yaml` file
that's in this directory to provide the path to your dataset to use for
calibration and evaluation.
```
old_path="/path/to/calibration/dataset"
sed -i "s|$old_path|$DATASET_DIR|g" inception_v3.yaml
old_path="/path/to/evaluation/dataset"
sed -i "s|$old_path|$DATASET_DIR|g" inception_v3.yaml
```

Run the tuning script:
```
bash run_tuning.sh --config=inception_v3.yaml \
  --input_model=inceptionv3_fp32_pretrained_model.pb \
  --output_model=${OUTPUT_DIR}/lpot_inceptionv3.pb
```

After this completes, the script outputs the results of the tuning and the
quantized model is saved to your output directory.
