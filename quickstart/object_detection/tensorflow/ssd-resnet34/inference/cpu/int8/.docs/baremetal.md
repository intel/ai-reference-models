<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git
* libgl1-mesa-glx
* libglib2.0-0
* numpy>=1.17.4
* Cython
* contextlib2
* pillow>=9.3.0
* lxml
* jupyter
* matplotlib
* pycocotools
* horovod==0.20.0
* tensorflow-addons==0.11.0
* opencv-python

In addition to the libraries above, <model name> uses the
[TensorFlow models](https://github.com/tensorflow/models) and
[TensorFlow benchmarks](https://github.com/tensorflow/benchmarks)
repositories. Clone the repositories using the commit ids specified
below and set the `TF_MODELS_DIR` to point to the folder where the models
repository was cloned:
```
# Clone the TensorFlow models repo
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
export TF_MODELS_DIR=$(pwd)
cd ..

# Clone the TensorFlow benchmarks repo
git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
cd ssd-resnet-benchmarks
git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
cd ..
```

Download the <model name> pretrained model for either the 300x300 or 1200x1200
input size, depending on which [quickstart script](#quick-start-scripts) you are
going to run. Set the `PRETRAINED_MODEL` environment variable for the path to the
pretrained model that you'll be using.
```
# ssd-resnet34 300x300
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_bs1_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_int8_bs1_pretrained_model.pb

# ssd-resnet34 1200x1200
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_1200x1200_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/ssd_resnet34_int8_1200x1200_pretrained_model.pb
```

Set an environment variable for the path to an `OUTPUT_DIR`
where log files will be written. If the accuracy test is being run, then
also set the `DATASET_DIR` to point to the folder where the COCO dataset
`validation-00000-of-00001` file is located. Once the environment
variables are setup, then run a [quickstart script](#quick-start-scripts).

To run inference using synthetic data:
```
export OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/int8_inference.sh
```

To test accuracy using the COCO dataset:
```
export DATASET_DIR=<path to the coco directory>
export OUTPUT_DIR=<directory where log files will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/int8_accuracy.sh
```
