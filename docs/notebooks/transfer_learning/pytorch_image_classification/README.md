# Transfer Learning for Image Classification using PyTorch

This notebook uses image classification models from torchvision that were originally trained 
using ImageNet and does transfer learning with the Food101 dataset, a flowers dataset, or
a custom image dataset.

The notebook performs the following steps:

1. Import dependencies and setup parameters
2. Prepare the dataset
3. Predict using the original model
4. Transfer learning
5. Visualize the model output
6. Export the saved model


## Running the notebook

The instructions below explain how to run the notebook on bare metal using a
virtual environment. It is assumed that the host has Python3 and wget pre-installed.

1. Get a clone of the Model Zoo repository from GitHub:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Create a Python3 virtual environment and install the required packages.
   
   You can use virtualenv:
   ```
   python3 -m venv intel-pyt-venv
   source intel-pyt-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n intel-pyt python=3.9
   conda activate intel-pyt
   ```
   Then, from inside the activated virtualenv or conda environment run these steps:
   ```
   pip install --upgrade pip
   pip install -r ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/pytorch_image_classification/requirements.txt
   ```
3. Set environment variables for the path to the dataset folder and an output directory.
   The dataset and output directories can be empty. The notebook will download the
   dataset to the dataset directory, if it is empty. Subsequent runs will reuse the dataset.
   ```
   export DATASET_DIR=<directory to download the dataset>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server)
   (note: we recommend using the increased `iopub_msg_rate_limit` and `iopub_data_rate_limit` values
   when using torchvision's model and data download APIs):
   ```
   cd ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/pytorch_image_classification
   jupyter notebook --NotebookApp.iopub_msg_rate_limit=10000.0 --NotebookApp.iopub_data_rate_limit=1e10 --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run
   the notebook.
   
Dataset Citations:

@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}

@ONLINE {tfflowers,
author = "The TensorFlow Team",
title = "Flowers",
month = "jan",
year = "2019",
url = "http://download.tensorflow.org/example_images/flower_photos.tgz" }

