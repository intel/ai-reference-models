<!--- 30. Datasets -->
## Datasets

Decide the problem you want to run to get the appropriate dataset.
We will get the training data of it as an example the computing BLEU score:
```
export DATASET_DIR=/home/<user>/transformer_data
mkdir $DATASET_DIR && cd $DATASET_DIR
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
```
Download training/evaluation datasets using [data_download.py script](https://github.com/mlcommons/training/blob/master/translation/tensorflow/transformer/data_download.py) 
```
python data_download.py --data_dir=$DATASET_DIR
```
Once the script completes, you should have a dataset directory `DATASET_DIR` with
the following files: `newstest2014.de`, `newstest2014.en`, and
a vocab text file `vocab.ende.32768`.

More detailed instructions about the model and preparing the dataset can be found [here]((https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)).
