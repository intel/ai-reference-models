# ImageNet Dataset Scripts

## Download ImageNet Dataset

[ImageNet]: https://image-net.org
[ILSVRC2012_devkit_t12.tar.gz]: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
[ILSVRC2012_img_train.tar]: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
[ILSVRC2012_img_val.tar]: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

[extract_ILSVRC.sh]: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
[torchvision.datasets.ImageNet]: https://pytorch.org/vision/0.16/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet
[torchvision.datasets.ImageFolder]: https://pytorch.org/vision/0.16/generated/torchvision.datasets.ImageFolder.html

[TFRecord]: https://www.tensorflow.org/tutorials/load_data/tfrecord

To download [ImageNet] datasets visit https://image-net.org, log in (or create an account if you don't have one) and request access to the datasets. Once access is approved go to the "Download" page and download 3 files for ImageNet 2012 dataset:

* [ILSVRC2012_devkit_t12.tar.gz] - 2.5MB (MD5: `fa75699e90414af021442c21a62c3abf`)
* [ILSVRC2012_img_train.tar] - training images, 138GB (MD5: `1d675b47d978889d74fa0da5fadfb00e`)
* [ILSVRC2012_img_val.tar] - validation images, 6.3GB (MD5: `29b22e2961454d5413ddabcf34fc5622`)

ImageNet datasets might require pre-processing before they will be used with AI frameworks. See sections below for details.

## Pre-process for PyTorch*

If sample is using [torchvision.datasets.ImageNet] API, then there is no need to preprocess ImageNet dataset before running the sample. On the first run [torchvision.datasets.ImageNet] will extract archives and place content appropriately. Consequent runs will skip extraction. After extraction you should see the following file structure:
```
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
```

If sample is using generic [torchvision.datasets.ImageFolder] API, then ImageNet dataset should be pre-processed. For this purpose use [extract_ILSVRC.sh] PyTorch example script (**note:** this script will remove input `*.tar` files).

Refer to sample documentation for the exact steps to pre-process ImageNet dataset.

## Pre-process for TensorFlow*

Tensorflow requires conversion of the dataset to [TFRecord] format.

1. Setup a python virtual environment with TensorFlow and other dependencies specified below:
   ```
   python3 -m venv tf_env
   source tf_env/bin/activate
   pip install --upgrade pip
   pip install intel-tensorflow
   pip install -I urllib3
   pip install wget
   ```

1. Download and run the [imagenet_to_tfrecords.sh](imagenet_to_tfrecords.sh) script passing a path to the directory where ImageNet `*.tar` files were downloaded:
   ```
   wget https://raw.githubusercontent.com/IntelAI/models/master/datasets/imagenet/imagenet_to_tfrecords.sh

   # To pre-process only the validation dataset:
   ./imagenet_to_tfrecords.sh $IMAGENET_DIR inference

   # To pre-process the entire dataset:
   ./imagenet_to_tfrecords.sh $IMAGENET_DIR training
   ```

The `imagenet_to_tfrecords.sh` script will extract the ImageNet tar files, download and then run the [`imagenet_to_gcs.py`](imagenet_to_gcs.py) script to convert the dataset images to [TFRecord]. As the script is running you should see output like this:
```
I0911 16:23:59.174904 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/train/train-00000-of-01024
I0911 16:23:59.199399 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/train/train-00001-of-01024
I0911 16:23:59.221770 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/train/train-00002-of-01024
I0911 16:23:59.251754 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/train/train-00003-of-01024
...
I0911 16:24:22.338566 140581751400256 imagenet_to_gcs.py:402] Processing the validation data.
I0911 16:24:23.271091 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00000-of-00128
I0911 16:24:24.260855 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00001-of-00128
I0911 16:24:25.179738 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00002-of-00128
I0911 16:24:26.097850 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00003-of-00128
I0911 16:24:27.028785 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00004-of-00128
...
```

After the `imagenet_to_gcs.py` script completes, the `imagenet_to_tfrecords.sh` script moves the train and validation files into the `$IMAGENET_DIR/tf_records` directory. The folder should contain 1024 training files and 128 validation files:
```
$ ls -1 $IMAGENET DIR/tf_records/
train-00000-of-01024
train-00001-of-01024
train-00002-of-01024
train-00003-of-01024
...
validation-00000-of-00128
validation-00001-of-00128
validation-00002-of-00128
validation-00003-of-00128
...
```
