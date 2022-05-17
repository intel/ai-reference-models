## Datasets

<model name> <mode> uses the config file and enwiki-20200101 dataset from the
[MLCommons training GitHub repo](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert).

Follow the instructions in their documentation to download the files and
preprocess the dataset to create TF records files. Set the `DATASET_DIR`
environment variable to the path to the TF records directory. Your directory
should look similar like this:
```
<DATASET_DIR>
├── seq_128
│   └── part-00000-of-00500_128
└── seq_512
    └── part-00000-of-00500
```

Download the `bert_config.json` file from the Google drive that is linked at the
[MLCommons BERT README](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#location-of-the-input-files).
Set the `CONFIG_FILE` environment variable the path to the downloaded file
when running the phase 1 quickstart script.
```
export CONFIG_FILE=<path to bert_config.json>
```
