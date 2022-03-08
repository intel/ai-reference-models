# Transformer Language MLPerf Dataset

* Download the newstest2014 dataset for computing BLEU score from [WMT'14 English-German data](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/).
```
export DATASET_DIR=/home/<user>/transformer_data
mkdir $DATASET_DIR && cd $DATASET_DIR
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
```

* Use [`data_download.py`](https://github.com/IntelAI/models/blob/master/models/language_translation/tensorflow/transformer_mlperf/training/fp32/transformer/data_download.py) script to download and preprocess WMT training and evaluation datasets.
The Model Zoo directory comes with [AI Kit](/docs/general/tensorflow/AIKit.md). If
you are not using AI kit, you will need a clone of the [Model Zoo repository](https://github.com/IntelAI/models).

* Running the `data_download.py` script assumes you have a Python environment with TensorFlow installed.
Create a virtual environment and install the following dependencies, then run the script.
    ```
    python3 -m venv tf_env
    source tf_env/bin/activate
    pip install --upgrade pip
    pip install intel-tensorflow
    
    export PYTHONPATH=$PYTHONPATH:<model zoo repo dir>/models/common/tensorflow
    export DATASET_DIR=/home/<user>/transformer_data
    
    cd <model zoo repo dir>/models/language_translation/tensorflow/transformer_mlperf/training/fp32/transformer
    python data_download.py --data_dir=$DATASET_DIR
    ```
    Once the script completes, you should have a dataset directory `DATASET_DIR` with the following files: `newstest2014.de`, `newstest2014.en`,
    a vocab text file `vocab.ende.32768`, and the preprocessed TFRecord data.
