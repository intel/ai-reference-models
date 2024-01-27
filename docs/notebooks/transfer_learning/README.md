# Transfer Learning Notebooks

## Environment setup and running the notebooks

Use the [setup instructions](setup.md) to install the dependencies required to run the notebooks.

This directory has Jupyter notebooks that demonstrate transfer learning.
All of the notebooks use models from public model repositories
and leverage optimized libraries [ Intel® Optimization for TensorFlow](https://pypi.org/project/intel-tensorflow/)
and [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

## Natural Language Processing

| Notebook | Domain: Use Case | Framework| Description |
| ---------| ---------|----------|-------------|
| [BERT SQuAD fine tuning with TF Hub](/docs/notebooks/transfer_learning/text_classification/bert_classifier_fine_tuning) | NLP: Question Answering | TensorFlow | Demonstrates BERT fine tuning using scripts from the [TensorFlow Model Garden](https://github.com/tensorflow/models) and the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). The notebook allows for selecting a BERT large or BERT base model from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [BERT Text Classification with TF Hub](/docs/notebooks/transfer_learning/text_classification/tfhub_bert_text_classification/) | NLP: Text Classification | TensorFlow | Demonstrates BERT binary text classification fine tuning using the [IMDb movie review dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) and multiclass text classification fine tuning using the [AG News datasets](https://www.tensorflow.org/datasets/catalog/ag_news_subset) from [TensorFlow Datasets](https://www.tensorflow.org/datasets) or a custom dataset (for binary classification). The notebook allows for selecting a BERT encoder (BERT large, BERT base, or small BERT) to use along with a preprocessor from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [Text Classifier fine tuning with PyTorch & Hugging Face](/docs/notebooks/transfer_learning/text_classification/pytorch_text_classification) | NLP: Text Classification | PyTorch |Demonstrates fine tuning [Hugging Face models](https://huggingface.co/models) to do sentiment analysis using the [IMDb movie review dataset from Hugging Face Datasets](https://huggingface.co/datasets/imdb) or a custom dataset with [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) |
| [LLM instruction tuning with PyTorch & Hugging Face](/docs/notebooks/transfer_learning/text_generation/pytorch_text_generation) | NLP: Text Generation | PyTorch | Demonstrates instruction tuning [Hugging Face Causal Language Models](https://huggingface.co/docs/transformers/tasks/language_modeling) to do text generation using [Hugging Face Datasets](https://huggingface.co/datasets/imdb) or a custom dataset |


## Computer Vision

| Notebook | Domain: Use Case | Framework | Description |
| ---------| ----------| ----------|-------------|
| [Image Classification with TF Hub](/docs/notebooks/transfer_learning/image_classification/tf_image_classification) | CV: Image Classification | TensorFlow | Demonstrates transfer learning with multiple [TF Hub](https://tfhub.dev) image classifiers, TF datasets, and custom image datasets |
| [Image Classification with PyTorch & Torchvision](/docs/notebooks/transfer_learning/image_classification/pytorch_image_classification) | CV: Image Classification | PyTorch | Demonstrates transfer learning with multiple [Torchvision](https://pytorch.org/vision/stable/index.html) image classification models, Torchvision datasets, and custom datasets |
| [Image Classification with Hugging Face](/docs/notebooks/transfer_learning/image_classification/huggingface_image_classification) | CV: Image Classification | PyTorch, TensorFlow | Demonstrates transfer learning with [Hugging Face models](https://huggingface.co/models) for image classification using either TensorFlow or PyTorch as a backend. |
| [Object Detection with PyTorch & Torchvision](/docs/notebooks/transfer_learning/object_detection/pytorch_object_detection) | CV: Object Detection | PyTorch |Demonstrates transfer learning with multiple [Torchvision](https://pytorch.org/vision/stable/index.html) object detection models, a public image dataset, and a customized Torchvision dataset |
