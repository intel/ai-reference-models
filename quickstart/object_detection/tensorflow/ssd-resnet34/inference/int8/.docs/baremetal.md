<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==2.3.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git
* libgl1-mesa-glx
* libglib2.0-0
* numpy==1.17.4
* Cython
* contextlib2
* pillow>=7.1.0
* lxml
* jupyter
* matplotlib
* pycocotools
* horovod==0.20.0
* tensorflow-addons==0.8.1
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
$ git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
$ cd ..
```

After installing the prerequisites and cloning the required repositories,
download and untar the model package. The model package includes the
<model name> <precision> pretrained model and the scripts needed to run
<mode>.
```
wget <package url>
tar -xzf <package name>
cd <package dir>
```

Set an environment variable for the path to an `OUTPUT_DIR`
where log files will be written. If the accuracy test is being run, then
also set the `DATASET_DIR` to point to the folder where the COCO dataset
`validation-00000-of-00001` file is located. Once the environment
variables are setup, then run a [quickstart script](#quick-start-scripts).

To run inference using synthetic data:
```
export OUTPUT_DIR=<directory where log files will be written>

quickstart/int8_inference.sh
```

To test accuracy using the COCO dataset:
```
export DATASET_DIR=<path to the coco directory>
export OUTPUT_DIR=<directory where log files will be written>

quickstart/int8_accuracy.sh
```
