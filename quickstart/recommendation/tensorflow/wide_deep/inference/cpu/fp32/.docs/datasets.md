<!--- 30. Datasets -->
## Dataset
Download and preprocess the [income census data](https://archive.ics.uci.edu/ml/datasets/Census+Income) by running 
following python script, which is a standalone version of [census_dataset.py](https://github.com/tensorflow/models/blob/master/official/wide_deep/census_dataset.py) Please note that below program requires `requests` module to be installed. You can install is using `pip install requests`.
Dataset will be downloaded in directory provided using `--data_dir`. If you are behind proxy then you can proxy urls
using `--http_proxy` and `--https_proxy` arguments.
```
$ git clone https://github.com/IntelAI/models.git
$ cd models
$ python ./benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/data_download.py --data_dir /home/<user>/widedeep_dataset
```
