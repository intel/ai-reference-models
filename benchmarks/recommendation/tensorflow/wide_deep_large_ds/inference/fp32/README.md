<!--- 0. Title -->
# Wide and Deep using a large dataset FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running Wide and Deep using a large dataset FP32 inference using
Intel-optimized TensorFlow.

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
               <docker image> \
               /bin/bash

    ```
* Now run the data preprocessing step:
    ```
     cd /dataset
     python /workspace/<package dir>/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
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
| [`fp32_online_inference.sh`](fp32_online_inference.sh) | Runs online inference (`batch_size=1`). The `NUM_OMP_THREADS` environment variable and the hyperparameters `num-intra-threads`, `num-inter-threads` can be tuned for best performance. If `NUM_OMP_THREADS` is not set, it will default to `1`. |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Measures the model accuracy (`batch_size=1000`). |

<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>AI Kit does not currently support TF 1.15.2 models</p>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3.6 or 3.7
        <li><a href="https://pypi.org/project/intel-tensorflow/1.15.2/">intel-tensorflow==1.15.2</a>
        <li>numactl
        <li>git
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After completing the setup, download the pretrained model frozen graph
and save the path to the PRETRAINED_MODEL envionment variable.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.pb
export PRETRAINED_MODEL=$(pwd)/wide_deep_fp32_pretrained_model.pb
```

Set environment variables for the path to your TF records file and an
`OUTPUT_DIR` where log files will be written, then navigate to your model
zoo directory and run a [quickstart script](#quick-start-scripts).

```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to TF records file>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph>

# Run inference with an accuracy check
./quickstart/recommendation/tensorflow/wide_deep_large_ds/inference/cpu/fp32/fp32_accuracy.sh

# Run online inference and set NUM_OMP_THREADS
export NUM_OMP_THREADS=1
./quickstart/recommendation/tensorflow/wide_deep_large_ds/inference/cpu/fp32/fp32_online_inference.sh \
--num-intra-threads 1 --num-inter-threads 1
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/wide-deep-large-dataset-fp32-inference-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/wide-deep-large-dataset-fp32-inference-tensorflow-container.html).

