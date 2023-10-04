<!--- 0. Title -->
# Vision Transformer Training

<!-- 10. Description -->

### Description
This document has instructions for running Vision Transformer training (fine tuning ) for FP32/ FP16 / BFloat16 datatypes. The model based on this [paper](https://arxiv.org/abs/2010.11929).
The script can be used to perform fine tuning for image recognition task on downstream dataset.
Script uses, HuggingFace [TFViTForImageClassification API](https://huggingface.co/docs/transformers/model_doc/vit#transformers.TFViTForImageClassification). 
For initial checkpoint you can use a pretrained model from https://huggingface.co/ , for example : https://huggingface.co/google/vit-base-patch16-224-in21k

### Downloading Pre-trained model
Downlad a pre-trained model from HuggingFace models repo, for example Vision transformer trained on imagenet 21k : https://huggingface.co/google/vit-base-patch16-224-in21k/tree/main
Make sure to download 'tf_model.h5' (model file) and 'config.json' (model configuration file)
Save tf_model.h5 and config.json in 'my_directory'.
Set 'my_directory' as INIT_CHECKPOINT_DIR.

##Note:
Make sure your add "id2label" and "label2id" fields in config.json file based on the dataset you will be using to fine-tune pretrained model.
For ex: 
refer to 'config.json' file at 'models/image_recognition/tensorflow/vision_transformer/training/' in this repo for imagenet2012 dataset with "id2label" and "label2id" specific to imagenet2012 labels
This config.json should be used along with 'tf_model.h5' to fine-tune [vit-base-patch16-224-in21k]( https://huggingface.co/google/vit-base-patch16-224-in21k) on imagenet2012.


### Dataset

Download and preprocess the ImageNet dataset using the [instructions here](/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running Vision Transformer.

## Run the model

### Run on Linux
```
For the environment setup , make sure you have following python packages install in your python3 environment:
tensorflow
numpy
transformers


# cd to your model zoo directory
cd model_zoo

To install the exact versions of the packages required (as validated before) for the model please run the following:
```
pip install -r ./models/image_recognition/tensorflow/vision_transformer/training/requirements.txt
```

export INIT_CHECKPOINT_DIR=<path to initial checkpoint or pretrained model>
export DATASET_DIR=<path to the ImageNet TF records>
export PRECISION=<set the precision to "fp32", "bfloat16" or "fp16">
export OUTPUT_DIR=<path to the directory where log files and checkpoints will be written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value of 512.
export BATCH_SIZE=<customized batch size value>

./quickstart/image_recognition/tensorflow/vision_transformer/training/cpu/run_vit_fine_tune.sh
```
Above script will run the training on all the cpu nodes available.

### Run on Linux with multi-instance
Running on multiple sockets/ numa nodes might not give optimum results due to inter-socket communication overhead.
For better result, we should run multiple instances of training, with each instance run on single numa node

To Run the multi-instance training, that is, run one instance of training per numa node, use the same setting/ environment variables as above and run the following script, which uses **openmpi** to run parallel instances
```
./quickstart/image_recognition/tensorflow/vision_transformer/training/cpu/run_vit_fine_tune_multiinstance.sh
```
