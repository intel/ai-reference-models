<!--- 0. Title -->
# Wide and Deep using a large dataset FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running Wide and Deep using a large dataset FP32 inference using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[wide-deep-large-ds-fp32-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/wide-deep-large-ds-fp32-inference.tar.gz)

<!--- 30. Datasets -->
## Datasets

The large [Kaggle Display Advertising Challenge Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
will be used. The
[data](https://www.kaggle.com/c/criteo-display-ad-challenge/data) is from
[Criteo](https://www.criteo.com) and has a field indicating if an ad was
clicked (1) or not (0), along with integer and categorical features.

Download large Kaggle Display Advertising Challenge Dataset from
[Criteo Labs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
* Download the large version of evaluation dataset from: https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
* Download the large version of train dataset from: https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv

Follow the instructions to convert the downloaded dataset to tfrecords using [preprocess_csv_tfrecords.py](/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py):
* Store the path to 
    ```
    mkdir dataset
    cd /home/<user>/dataset
    ```

Copy the eval.csv and test.csv into your current working directory `/home/<user>/dataset`

* Launch docker 
    ```
    cd /home/<user>/dataset
    docker run -it --privileged -u root:root \
               --volume /home/<user>/dataset:/dataset \
               intel/recommendation:tf-1.15.2-imz-2.2.0-wide-deep-large-ds-fp32-inference \
               /bin/bash

    ```
* Now run the data preprocessing step:
    ```
     cd /dataset
     python /workspace/wide-deep-large-ds-fp32-inference/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
         --inputcsv-datafile eval.csv \
         --calibrationcsv-datafile train.csv \
         --outputfile-name preprocessed_eval
    ```
Now preprocessed eval dataset will be stored as eval_preprocessed_eval.tfrecords in /home/<user>/dataset directory.

* Exit out of docker once the dataset pre-processing completes.
    ```
    exit
    ```

Set the `DATASET_DIR` to point to this directory when running Wide and Deep using a large dataset:
```
export DATASET_DIR=/home/<user>/dataset/eval_preprocessed_eval.tfrecords
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](fp32_online_inference.sh) | Runs online inference (`batch_size=1`). The `NUM_OMP_THREADS` environment variable and the hyperparameters `num-intra-threads`, `num-inter-threads` can be tuned for best performance. |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Measures the model accuracy (`batch_size=1000`). |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/)
* numactl

After installing the prerequisites, download and untar the model package.
Set environment variables for the path to your `DATASET_DIR` and an
`OUTPUT_DIR` where log files will be written, then run a 
[quickstart script](#quick-start-scripts).

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/wide-deep-large-ds-fp32-inference.tar.gz
tar -xzf wide-deep-large-ds-fp32-inference.tar.gz
cd wide-deep-large-ds-fp32-inference
```

* Running inference to check accuracy:
```
quickstart/fp32_accuracy.sh
```
* Running online inference:
Set `NUM_OMP_THREADS` for tunning the hyperparameter `num_omp_threads`.
```
NUM_OMP_THREADS=1
quickstart/fp32_online_inference.sh \
--num-intra-threads 1 --num-inter-threads 1
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
Wide and Deep using a large dataset FP32 inference. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset
and an output directory.

* Running inference to check accuracy:
```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/recommendation:tf-1.15.2-imz-2.2.0-wide-deep-large-ds-fp32-inference \
  /bin/bash quickstart/fp32_accuracy.sh
```

* Running online inference:
Set `NUM_OMP_THREADS` for tunning the hyperparameter `num_omp_threads`.

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log files will be written>
NUM_OMP_THREADS=1

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env NUM_OMP_THREADS=${NUM_OMP_THREADS} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -t \
  intel/recommendation:tf-1.15.2-imz-2.2.0-wide-deep-large-ds-fp32-inference \
  /bin/bash quickstart/fp32_online_inference.sh \
  --num-intra-threads 1 --num-inter-threads 1
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

