# Fine-tuning using BERT models from TF Hub

These notebooks demonstrate fine tuning using various [BERT](https://arxiv.org/abs/1810.04805) models
from [TF Hub](https://tfhub.dev) using Intel® Optimization for TensorFlow. There are notebooks for the
following tasks:

| Notebook | Description |
|----------|-------------|
| [BERT_Question_Answering.ipynb](BERT_Question_Answering.ipynb) | Demonstrates BERT fine tuning using scripts from the [TensorFlow Model Garden](https://github.com/tensorflow/models) and the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). The notebook allows for selecting a BERT large or BERT base model from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [BERT_Binary_Text_Classification.ipynb](BERT_Binary_Text_Classification.ipynb) | Demonstrates BERT binary text classification fine tuning using the [IMDb movie review dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) from [TensorFlow Datasets](https://www.tensorflow.org/datasets) or a custom dataset. The notebook allows for selecting a BERT encoder (BERT large, BERT base, or small BERT) to use along with a preprocessor from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |

## Running the notebooks

The instructions below explain how to run the notebooks on [bare metal](#bare-metal) using a
virtual environment or in a [docker container](#using-docker). The notebook itself will do additional
installations for dependencies that it needs, but it will assume that TensorFlow has already
been installed.

### Bare Metal

1. Get a clone of the Model Zoo for Intel Architecture repository or download the notebooks from GitHub:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Create a Python3 virtual environment and install `intel-tensorflow` and `notebook`:
   ```
   python3 -m venv intel-tf-venv
   source intel-tf-venv/bin/activate
   pip install --upgrade pip
   pip install intel-tensorflow==2.8.0 notebook
   ```
3. Set environment variables for the path to the working folder and an output directory. Both folders
   need to be writable. The working directory is used for downloading dependencies needed to run
   the model, including datasets, vocab files, and config files.
   ```
   export WORKING_DIR=<workspace used for datasets and other dependencies>
   export OUTPUT_DIR=<output directory for checkpoints and the saved model>

   mkdir -p $WORKING_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo (or the directory
   where you downloaded the notebooks), and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/tfhub_bert
   jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run
   the notebook.

### Using Docker

1. Get a clone of the Model Zoo for Intel Architecture repository or download the notebooks from GitHub:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Use the `intel/intel-optimized-tensorflow:jupyter` docker container to
   run the notebook.

   The snippet below shows how to mount directories in the container for your clone
   of the model zoo, a working directory (used to download the dataset, the vocab file,
   the bert config, etc), and a directory for output (like checkpoints and the saved
   model).
   ```
   WORKING_DIR=<workspace used for datasets and other dependencies>
   OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $WORKING_DIR
   mkdir -p $OUTPUT_DIR

   # The notebook directory in your clone of the model zoo, or your notebook download directory
   NOTEBOOK_DIR=${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/tfhub_bert

   docker run -d --rm \
     -p 8888:8888 \
     --name intel-tensorflow-jupyter \
     --env http_proxy=${https_proxy} \
     --env https_proxy=${https_proxy} \
     --env WORKING_DIR=${WORKING_DIR} \
     --env OUTPUT_DIR=${OUTPUT_DIR} \
     -v ${WORKING_DIR}:${WORKING_DIR} \
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

### References

Dataset citations:
```
@article{2016arXiv160605250R,
       author = { {Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}

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

@misc{misc_sms_spam_collection_228,
  author       = {Almeida, Tiago},
  title        = {{SMS Spam Collection}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository}
}
```

TensorFlow Model Garden citation:
```
@misc{tensorflowmodelgarden2020,
  author = {Hongkun Yu and Chen Chen and Xianzhi Du and Yeqing Li and
            Abdullah Rashwan and Le Hou and Pengchong Jin and Fan Yang and
            Frederick Liu and Jaeyoun Kim and Jing Li},
  title = {{TensorFlow Model Garden}},
  howpublished = {\url{https://github.com/tensorflow/models}},
  year = {2020}
}
```
