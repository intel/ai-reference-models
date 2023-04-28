# Dataset API

## Dataset API structure
```
models/datasets/dataset_api/
|── dataset.py
├── dataset_urls.json
├── README.md
├── scripts
├── setup.sh
└── dataset_api/
    ├── __init__.py
    ├── download.py
    └── preprocess.py
```

## Environment setup
Clone the [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models) repository and navigate to the `dataset_api` directory.
```
# Optional: create and activate a virtual environment
virtualenv -p python3 venv
. venv/bin/activate

cd models/datasets/dataset_api
# Install dependencies, some dependencies might require root privilages.
./setup.sh
```

## Datasets
| Dataset name | Description | Download | Preprocessing | command |
| ------------ | ----------- | -------- | --------------| ------- |
| `brca` | [Breast Cancer dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#10937961150f24f71b869471e8366180549549d75) that contains categorized contrast enhanced mammography data and radiologists’ notes. | supported | **A prerequisite:** Use a browser, download [the Low Energy and Subtracted images](https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/260?passcode=5335d2514638afdaf03237780dcdfec29edf4238#), then provide the path to the directory that contains the downloaded images using `--directory` argument. | `python dataset.py -n brca --download --preprocess -d <path to the dataset directory>` |
| `tabformer` | [Credit card data](https://ibm.ent.box.com/v/tabformer-data/folder/130748337023) for TabFormer | supported | not supported | `python dataset.py -n tabformer --download` |
| `dureader-vis` | [DuReader-vis](https://github.com/baidu/DuReader/tree/master/DuReader-vis) for document automation. Chinese Open-domain Document Visual Question Answering (Open-Domain DocVQA) dataset, containing about 15K question-answering pairs and 158K document images from the Baidu search engine. | supported | not supported  | `python dataset.py -n dureader-vis --download` |
| `msmarco` | [MS MARCO](https://microsoft.github.io/msmarco/)  is a collection of datasets focused on deep learning in search | supported | not supported | `python dataset.py -n msmarco --download` |
| `mvtec-ad` | [MVTEC Anomaly Detection DATASET](https://www.mvtec.com/company/research/datasets/mvtec-ad) for industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. | supported | supported  | `python dataset.py -n mvtec-ad --download --preprocess -d <path to the dataset directory>` |

## Command-line Interface

| Input Arguments | Description |
| --------------- | ----------- |
| --list (-l) | list the supported datasets. |
| --name (-n) | dataset name |
| --directory (-d) | directory location where the raw dataset will be saved on your system. It's also where the preprocessed dataset files will be written. If not set, a directory with the dataset name will be created. |
| --download | download the dataset specified. |
| --preprocess | preprocess the dataset if supported. |


## Python API
```
from dataset_api.download import download_dataset
from dataset_api.preprocess import preprocess_dataset

# Download the datasets
download_dataset('brca', <path to the raw dataset directory>)

# Preprocess the datasets
preprocess_dataset('brca', <path to the raw dataset directory>)
```
