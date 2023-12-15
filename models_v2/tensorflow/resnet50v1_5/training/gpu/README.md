# ResNet50 Model Training Convergence for Intel® Extention for TensorFlow

## Model Information
| **Case** |**Framework** | **Model Repo** | **Tag** 
| :---: | :---: | :---: | :---: |
| Training | Tensorflow | [Tensorflow-Models](https://github.com/tensorflow/models) | v2.8.0 |

# Pre-Requisite
* Host has Intel® Data Center GPU Max
* Host has installed latest Intel® Data Center GPU Max Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Dataset 
Using TensorFlow Datasets.
`classifier_trainer.py`` supports ImageNet with [TensorFlow Datasets(TFDS)](https://www.tensorflow.org/datasets/overview) .

Please see the following [example snippet](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/scripts/download_and_prepare.py) for more information on how to use TFDS to download and prepare datasets, and specifically the [TFDS ImageNet readme](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.md) for manual download instructions.

Legacy TFRecords
Download the ImageNet dataset and convert it to TFRecord format. The following [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) and [README](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy) provide a few options.

> Note that the legacy ResNet runners, e.g. [resnet/resnet_ctl_imagenet_main.py](https://github.com/tensorflow/models/blob/v2.8.0/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py) require TFRecords whereas `classifier_trainer.py` can use both by setting the builder to 'records' or 'tfds' in the configurations.

## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/resnet50v1_5/training/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters
  There are several config yaml files in configure and hvd_configure folder. Set one of them as CONFIG_FILE, then model would correspondly run with `real data` or `dummy data`. Single-tile please use yaml file in configure folder. Distribute training please use yaml file in hvd_configure folder, `itex_bf16_lars.yaml`/`itex_fp32_lars.yaml` for HVD real data and `itex_dummy_bf16_lars.yaml`/`itex_dummy_fp32_lars.yaml` for HVD dummy data.
Export those parameters to script or environment.

    |   **Parameter**    | **export command**                                    |
    | :---: | :--- |
    |  **DATASET_DIR**   | `export DATASET_DIR=/the/path/to/dataset`    (if you choose dummy data, you can igore this parameter)          |
    |   **OUTPUT_DIR**   | `export OUTPUT_DIR=/the/path/to/output_dir`           |
    |   **MULTI_TILE**   | `export MULTI_TILE=False (False or True)`           |
    |   **CONFIG_FILE**   | `path/to/itex_xx.yaml  (dataset type and precision) ` |
6. Run `run_model.sh`

## Output

Output will typically look like:
```
I1101 12:22:02.439692 139875177744192 keras_utils.py:145] TimeHistory: 57.00 seconds, xxx examples/second between steps 0 and 100
I1101 12:22:51.165375 139875177744192 keras_utils.py:145] TimeHistory: 48.71 seconds, xxx examples/second between steps 100 and 200
I1101 12:23:39.856714 139875177744192 keras_utils.py:145] TimeHistory: 48.69 seconds, xxx examples/second between steps 200 and 300
I1101 12:24:28.548917 139875177744192 keras_utils.py:145] TimeHistory: 48.69 seconds, xxx examples/second between steps 300 and 400

```

Final results of the training run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: xxx
   unit: images/sec
```
