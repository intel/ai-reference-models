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

Set the `DATASET_DIR` to point to this directory when running <model name>:
```
export DATASET_DIR=/home/<user>/dataset/eval_preprocessed_eval.tfrecords
```
