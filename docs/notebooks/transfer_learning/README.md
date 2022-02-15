# Transfer Learning Notebooks

This directory has Jupyter notebooks that demonstrate transfer learning with
models from the Model Zoo for Intel Architecture and other public model repositories using Intel TensorFlow.

| Notebook | Description |
| ---------| ------------|
| [BERT Classifier fine tuning using the Model Zoo for Intel Architecture](/docs/notebooks/transfer_learning/bert_classifier_fine_tuning/) | Fine tunes BERT base from the Model Zoo for Intel Architecture using the IMDB movie review dataset, then quantizes the saved model using the [Intel® Neural Compressor](https://github.com/intel/neural-compressor) |
| [BERT_SQuAD fine tuning using TF Hub](/docs/notebooks/transfer_learning/bert_tfhub/) | The notebook demonstrates BERT fine tuning using scripts from the [TensorFlow Model Garden](https://github.com/tensorflow/models) and the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). The notebook allows for selecting a BERT large or BERT base model from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [BERT Binary Text_Classification using TF Hub](/docs/notebooks/transfer_learning/bert_tfhub) | The notebook demonstrates BERT binary text classification fine tuning using the [IMDb movie review dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) from [TensorFlow Datasets](https://www.tensorflow.org/datasets). The notebook allows for selecting a BERT encoder (BERT large, BERT base, or small BERT) to use along with a preprocessor from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [Transfer Learning for Food-101 Image Classification](/docs/notebooks/transfer_learning/tfhub_classifier_food101) | Demonstrates transfer learning with a [TF Hub](https://tfhub.dev) image classifier and the Food-101 dataset |
| [Transfer Learning for General Object Detection](/docs/notebooks/transfer_learning/pytorch_object_detection) | Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) object detection models and a public image dataset |
 
