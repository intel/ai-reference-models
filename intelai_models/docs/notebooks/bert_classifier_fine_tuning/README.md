# BERT Classifier fine tuning using IMDB

This notebook uses the BERT BFloat16 classifier training scripts from the model zoo to
do fine tuning. The [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
is used to do sentiment analysis on movie reviews. For more information on the
large movie review dataset, please see the
[ACL 2011 paper](https://aclanthology.org/P11-1015/).

Steps that the notebook follows:
* Downloads the IMDB dataset
* Converts the dataset to .tsv files
* Downloads the BERT pretrained model checkpoints
* Runs BERT classifier fine tuning using the IMDB tsv files
* Exports the saved model
* Loads the saved model to test predictions

## Running the notebook

1. Since this notebook scripts from the Model Zoo, clone the repo:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Use the `intel/intel-optimized-tensorflow:jupyter` docker container to
   run the notebook.

   The snippet below shows how to mount directories in the container for the IMDB
   dataset, the BERT base checkpoint files, and a directory for output (logs, checkpoint
   files generated during training, and the saved model). The first time that the
   notebook is run, it will download the IMDB dataset to the `DATASET_DIR` and
   download the BERT base checkpoint files to the `CHECKPOINT_DIR`. The output
   directory should be empty when starting the notebook run, to prevent previously
   generated checkpoint files from getting picked up.
   ```
   DATASET_DIR=<directory to download the IMDB dataset>
   CHECKPOINT_DIR=<directory to download the BERT base initial checkpoints>
   OUTPUT_DIR=<empty output directory>

   mkdir -p $OUTPUT_DIR
   mkdir -p $CHECKPOINT_DIR
   mkdir -p $DATASET_DIR

   docker run -d --rm \
     -p 8888:8888 \
     --name intel-tensorflow-jupyter \
     --env http_proxy=$https_proxy \
     --env https_proxy=$https_proxy \
     --env DATASET_DIR=${DATASET_DIR} \
     --env OUTPUT_DIR=${OUTPUT_DIR} \
     --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
     --env MODEL_ZOO_DIR=${MODEL_ZOO_DIR} \
     -v ${DATASET_DIR}:${DATASET_DIR} \
     -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
     -v ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
     -v ${MODEL_ZOO_DIR}:${MODEL_ZOO_DIR} \
     -v ${MODEL_ZOO_DIR}/docs/notebooks/bert_classifier_fine_tuning:/tf \
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