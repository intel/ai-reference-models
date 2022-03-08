# Large Kaggle Display Advertising Challenge Dataset

The Large [Kaggle Display Advertising Challenge Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
will be used for training Wide and Deep. The [data](https://www.kaggle.com/c/criteo-display-ad-challenge/data) is from
[Criteo](https://www.criteo.com) and has a field indicating if an ad was
clicked (1) or not (0), along with integer and categorical features.

Download the Large Kaggle Display Advertising Challenge Dataset from [Criteo Labs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) in `$DOWNLOAD_DIR`.
If the evaluation/train dataset were not available in the above link, it can be downloaded as follow:
   ```
    export DOWNLOAD_DIR=<location where dataset files will be saved>
    mkdir $DOWNLOAD_DIR && cd $DOWNLOAD_DIR
    wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
    wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
   ```

Download and run the [preprocess_csv_tfrecords.py](https://github.com/IntelAI/models/tree/master/datasets/large_kaggle_advertising_challenge/preprocess_csv_tfrecords.py)
script to preprocess the dataset to TFrecords.
* Create a virtual environment and install dependencies to run the script:
   ```
         apt-get update
         apt-get install python-pandas
         virtualenv --python=python3.6 tf_env
         . tf_env/bin/activate
         pip install intel-tensorflow==1.15.2
         pip install pandas
   ```
    
* Now run the script for data preprocessing:
   ```
        cd $DOWNLOAD_DIR
        wget https://raw.githubusercontent.com/IntelAI/models/master/datasets/large_kaggle_advertising_challenge/preprocess_csv_tfrecords.py
        python preprocess_csv_tfrecords.py \
                --inputcsv-datafile eval.csv \
                --calibrationcsv-datafile train.csv \
                --outputfile-name preprocessed_eval
   ```
Now the preprocessed `eval and train` datasets will be stored as `eval_preprocessed_eval.tfrecords` in the `$DOWNLOAD_DIR` directory.
To run inference scripts, set `DATASET_DIR` environment variable to `$DOWNLOAD_DIR/eval_preprocessed_eval.tfrecords`.
