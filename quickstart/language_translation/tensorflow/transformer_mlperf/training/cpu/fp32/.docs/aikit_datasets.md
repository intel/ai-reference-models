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

For the training dataset, run the `data_download.py` script from the Model Zoo directory.
The Model Zoo directory comes with [AI Kit](/docs/general/tensorflow/AIKit.md). If
you are not using AI kit, you will need a clone of the Model Zoo repo.
```
export PYTHONPATH=$PYTHONPATH:<model zoo dir>/models/common/tensorflow
export DATASET_DIR=/home/<user>/transformer_data

cd <model zoo dir>/models/language_translation/tensorflow/transformer_mlperf/training/fp32/transformer
python data_download.py --data_dir=$DATASET_DIR
```

Running `python data_download.py --data_dir=$DATASET_DIR` assumes you have a python environment similar to what the `intel/intel-optimized-tensorflow:latest` container provides. One option would be to run the above within the `intel/intel-optimized-tensorflow:latest` container eg: `docker run -u $(id -u):$(id -g) --privileged  --entrypoint /bin/bash -v /home/<user>:/home/<user> -it intel/intel-optimized-tensorflow:latest`
