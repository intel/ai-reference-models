<!--- 30. Datasets -->
## Dataset
Download and preprocess the [income census data](https://archive.ics.uci.edu/ml/datasets/Census+Income) by running 
following python script, which is a standalone version of [census_dataset.py](https://github.com/tensorflow/models/blob/v2.2.0/official/r1/wide_deep/census_dataset.py)
Please note that below program requires `requests` module to be installed. You can install it using `pip install requests`.
Dataset will be downloaded in directory provided using `--data_dir`. If you are behind corporate proxy, then you can provide proxy URLs
using `--http_proxy` and `--https_proxy` arguments.
```
git clone https://github.com/IntelAI/models.git
cd models
python ./benchmarks/recommendation/tensorflow/wide_deep/inference/fp32/data_download.py --data_dir /home/<user>/widedeep_dataset
```
