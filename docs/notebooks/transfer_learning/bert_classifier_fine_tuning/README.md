# BERT Classifier fine tuning using IMDB

This notebook uses the BERT classifier training scripts from the model zoo to
do fine tuning using the [IMDB movie review sentiment analysis dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
After fine tuning, the [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
is used to do quantization.

For more information on the large movie review dataset, please see the
[ACL 2011 paper](https://aclanthology.org/P11-1015/).

Steps that the notebook follows:
* Downloads the IMDB dataset
* Converts the dataset to .tsv files
* Downloads the BERT pretrained model checkpoints
* Runs BERT classifier fine tuning using the IMDB tsv files
* Exports the saved model
* Loads the saved model to test predictions
* Uses the [Intel® Neural Compressor](https://github.com/intel/neural-compressor) for quantization

## Running the notebook

The instructions below explain how to run the notebook on [bare metal](#bare-metal) using a
virtual environment or in a [docker container](#using-docker).

### Bare Metal

1. Since this notebook requires scripts from the Model Zoo, clone the repo:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Create a Python3 virtual environment and install `intel-tensorflow` and
   the `notebook`:
   ```
   python3 -m venv intel-tf-venv
   source intel-tf-venv/bin/activate
   pip install --upgrade pip
   pip install intel-tensorflow notebook
   ```
3. Set environment variables for the path to the IMDB dataset folder, the BERT base
   initial checkpoint folder, and an empty output directory. The dataset and checkpoint
   files will be downloaded when the notebook is run, so the folders can be empty.
   ```
   export DATASET_DIR=<directory to download the IMDB dataset>
   export CHECKPOINT_DIR=<directory to download the BERT base initial checkpoints>
   export OUTPUT_DIR=<empty output directory>

   mkdir -p $OUTPUT_DIR
   mkdir -p $CHECKPOINT_DIR
   mkdir -p $DATASET_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo, and then
   start the [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/bert_classifier_fine_tuning
   jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run
   the notebook.

### Using Docker

1. Since this notebook requires scripts from the Model Zoo, clone the repo:
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
     -v ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/bert_classifier_fine_tuning:/tf \
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

### References

Dataset citation:
```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```