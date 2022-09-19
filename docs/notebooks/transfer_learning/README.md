# Transfer Learning Notebooks

This directory has Jupyter notebooks that demonstrate transfer learning with
models from the Model Zoo for Intel Architecture and other public model repositories using Intel-optimized TensorFlow
and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

## Natural Language Processing

| Notebook | Framework | Description |
| ---------| ----------|-------------|
| [BERT Classifier fine tuning using the Model Zoo for Intel Architecture](/docs/notebooks/transfer_learning/bert_classifier_fine_tuning/) | TensorFlow | Fine tunes BERT base from the Model Zoo for Intel Architecture using the IMDb movie review dataset, then quantizes the saved model using the [Intel® Neural Compressor](https://github.com/intel/neural-compressor) |
| [BERT SQuAD fine tuning with TF Hub](/docs/notebooks/transfer_learning/tfhub_bert/) | TensorFlow | Demonstrates BERT fine tuning using scripts from the [TensorFlow Model Garden](https://github.com/tensorflow/models) and the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). The notebook allows for selecting a BERT large or BERT base model from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [BERT Binary Text Classification with TF Hub](/docs/notebooks/transfer_learning/tfhub_bert) | TensorFlow | Demonstrates BERT binary text classification fine tuning using the [IMDb movie review dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) from [TensorFlow Datasets](https://www.tensorflow.org/datasets) or a custom dataset. The notebook allows for selecting a BERT encoder (BERT large, BERT base, or small BERT) to use along with a preprocessor from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [Text Classifier fine tuning with PyTorch & Hugging Face](/docs/notebooks/transfer_learning/pytorch_text_classification) | PyTorch | Demonstrates fine tuning [Hugging Face models](https://huggingface.co/models) to do sentiment analysis using the [IMDb movie review dataset from Hugging Face Datasets](https://huggingface.co/datasets/imdb) or a custom dataset with [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) |

## Computer Vision

| Notebook | Framework | Description |
| ---------| ----------|-------------|
| [Transfer Learning for Image Classification with TF Hub](/docs/notebooks/transfer_learning/tf_image_classification) | TensorFlow | Demonstrates transfer learning with multiple [TF Hub](https://tfhub.dev) image classifiers, TF datasets, and custom image datasets |
| [Transfer Learning for Image Classification with PyTorch & torchvision](/docs/notebooks/transfer_learning/pytorch_image_classification) | PyTorch | Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) image classification models, torchvision datasets, and custom datasets |
| [Transfer Learning for Object Detection with PyTorch & torchvision](/docs/notebooks/transfer_learning/pytorch_object_detection) | PyTorch | Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) object detection models, a public image dataset, and a customized torchvision dataset |

# Remove Notebook Cells and Run as a Script
Cells in a Jupyter notebook can be tagged with metadata and then selectively removed when the notebook
is converted into a python script via nbconvert with a custom preprocessor. This allows specific paths
through the notebook to be executed in automated tests. As an example, the
[Transfer Learning for Image Classification with TF Hub](/docs/notebooks/transfer_learning/tf_image_classification)
notebook can be exported and run with the following steps.

## Export and Run the Custom Dataset Code Path
1. Set up and activate the environment for the notebook according to its [README.md](/docs/notebooks/transfer_learning/tf_image_classification/README.md)
2. Export environment variables required by the notebook's [README.md](/docs/notebooks/transfer_learning/tf_image_classification/README.md) (e.g. DATASET_DIR and OUTPUT_DIR) 
3. Run the below command to convert the notebook into a script that can be executed by ipython (this assumes you are in the current directory):
```
jupyter nbconvert \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags remove_for_custom_dataset \
    --to script \
    tf_image_classification/Image_Classification_Transfer_Learning.ipynb
```
4. Run the exported script at the command line using `ipython`:
```
ipython tf_image_classification/Image_Classification_Transfer_Learning.py
```

## Define Other Custom Paths
To visually remove cells for a different code path, open the Jupyter notebook in a web browser, select "View" -> "Cell Toolbar", and then select either "Edit Metadata" or "Tags". Each cell of the notebook will then have a toolbar item available for adding or editing metadata tags. You will need to tag each cell you want to omit in the exported script with a unique string, and then run the nbconvert command with the new tag passed in place of `rm_custom_dataset` for the `--TagRemovePreprocessor.remove_cell_tags` argument.

