# Transfer Learning for Image Classification with TF Hub

This notebook uses transfer learning with multiple [TF Hub](https://tfhub.dev) image classifiers,
[TF datasets](https://www.tensorflow.org/datasets/), and custom image datasets.

Steps that the notebook follows:
* Install dependencies and setup parameters
* Prepare the dataset using either a TF dataset or your own images
* Predict using the original model
* Transfer learning
* Evaluate the model
* Export the saved model

## Running the notebook

The instructions below explain how to run the notebook on [bare metal](#bare-metal) using a
virtual environment or in a [docker container](#using-docker).

### Bare Metal

1. Get a clone of the Model Zoo for Intel Architecture repository from GitHub:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Create a Python3 virtual environment and install `intel-tensorflow` and `notebook`:
   ```
   python3 -m venv intel-tf-venv
   source intel-tf-venv/bin/activate
   pip install --upgrade pip
   pip install intel-tensorflow notebook
   ```
3. Set environment variables for the path to the dataset folder and an output directory.
   The dataset and output directories can be empty. The notebook will download the specified
   dataset to the dataset directory, if it is empty. Subsequent runs will reuse the dataset.
   ```
   export DATASET_DIR=<directory to the TF dataset or custom dataset>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/tf_image_classification
   jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run
   the notebook.

### Using Docker

1. Get a clone of the Model Zoo for Intel Architecture repository from GitHub:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Use the `intel/intel-optimized-tensorflow:jupyter` docker container to
   run the notebook.

   The snippet below shows how to mount directories in the container for your clone
   of the model zoo, a dataset directory, and a directory for output (like the saved
   model). The notebook will either download the specified TF dataset or use a custom
   dataset in the `DATASET_DIR`.
   ```
   DATASET_DIR=<directory to the TF dataset or custom dataset>
   OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR

   # The notebook directory in your clone of the model zoo
   NOTEBOOK_DIR=${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/tf_image_classification

   docker run -d --rm \
     -p 8888:8888 \
     --name intel-tensorflow-jupyter \
     --env http_proxy=$https_proxy \
     --env https_proxy=$https_proxy \
     --env DATASET_DIR=${DATASET_DIR} \
     --env OUTPUT_DIR=${OUTPUT_DIR} \
     -v ${DATASET_DIR}:${DATASET_DIR} \
     -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
     -v ${NOTEBOOK_DIR}:/tf \
     intel/intel-optimized-tensorflow:jupyter
   ```
3. After the container starts, view the logs to get the URL and token for
   the notebook. Copy the URL with the token to view the notebook into your
   browser to view your notebook (for example:
   `http://127.0.0.1:8888/?token=<token>`).
   ```
   docker logs intel-tensorflow-jupyter
   ```
4. When you are done using the notebook, you can stop the container:
   ```
   docker stop intel-tensorflow-jupyter
   ```

## Dataset citations
```
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
```
