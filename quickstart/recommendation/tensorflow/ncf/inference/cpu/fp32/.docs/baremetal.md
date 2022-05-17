<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* numactl
* google-api-python-client==1.6.7
* google-cloud-bigquery==0.31.0
* kaggle==1.3.9
* numpy==1.16.3
* oauth2client==4.1.2
* pandas
* 'psutil>=5.6.7'
* py-cpuinfo==3.3.0
* typing
* TensorFlow models, clone the official `tensorflow/models` repository with  tag `v1.11`:
```
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
git checkout v1.11
```

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>
TF_MODELS_DIR=<path to the TensorFlow models directory tf_models>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ncf-fp32-inference.tar.gz
tar -xzf ncf-fp32-inference.tar.gz
cd ncf-fp32-inference

./quickstart/<script name>.sh
```
