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
| [Transfer Learning for Object Detection with PyTorch & torchvision](/docs/notebooks/transfer_learning/pytorch_object_detection) | PyTorch | Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) object detection models and a public image dataset |
