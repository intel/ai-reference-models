<!--- 30. Datasets -->
## Datasets

Decide the problem you want to run to get the appropriate dataset.
We will get the training data of it as an example:

Download dataset for computing BLEU score
```
export DATASET_DIR=/home/<user>/transformer_data
mkdir $DATASET_DIR && cd $DATASET_DIR
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
```

For the training dataset, download and untar the model package.    
```
wget <package url>
tar -xzf <package name>

export PYTHONPATH=$PYTHONPATH:/home/<user>/<package dir>/models/common/tensorflow
export DATASET_DIR=/home/<user>/transformer_data
    
cd <package dir>/models/language_translation/tensorflow/transformer_mlperf/training/bfloat16/transformer
python data_download.py --data_dir=$DATASET_DIR
```

Running `python data_download.py --data_dir=$DATASET_DIR` assumes you have a python environment similar to what the `intel/intel-optimized-tensorflow:ubuntu-18.04` container provides. One option would be to run the above within the `intel/intel-optimized-tensorflow:ubuntu-18.04` container eg: `docker run -u $(id -u):$(id -g) --privileged  --entrypoint /bin/bash -v /home/<user>:/home/<user> -it intel/intel-optimized-tensorflow:ubuntu-18.04`


