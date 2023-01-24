<!--- 30. Datasets -->
## Dataset

The Large [Kaggle Display Advertising Challenge Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
will be used for training Wide and Deep. The [data](https://www.kaggle.com/c/criteo-display-ad-challenge/data) is from
[Criteo](https://www.criteo.com) and has a field indicating if an ad was
clicked (1) or not (0), along with integer and categorical features.

Download the Large Kaggle Display Advertising Challenge Dataset from [Criteo Labs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) in `$DATASET_DIR`.
If the evaluation/train dataset were not available in the above link, it can be downloaded as follow:
   ```
    export DATASET_DIR=<location where dataset files will be saved>
    mkdir $DATASET_DIR && cd $DATASET_DIR
    wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
    wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
   ```
The `DATASET_DIR` environment variable will be used as the dataset directory when running [quickstart scripts](#quick-start-scripts).
