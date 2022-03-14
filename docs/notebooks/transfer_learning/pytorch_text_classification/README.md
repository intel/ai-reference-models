# Text Classifier fine tuning using IMDb with PyTorch

This notebook demonstrates fine tuning [pretrained models from Hugging Face](https://huggingface.co/models)
using text classification datasets from the [Hugging Face Datasets catalog](https://huggingface.co/datasets) or
a custom dataset. The [IMDb Larget Movie Review dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is used
from the Hugging Face Datasets catalog, and the [SMS Spam Collection dataset](https://archive-beta.ics.uci.edu/ml/datasets/sms+spam+collection)
is used as an example of a custom dataset being loaded from a csv file.

The notebook uses
[Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) which extends PyTorch
with optimizations for extra performance boost on Intel hardware.

The notebook performs the following steps:
1. Import dependencies and setup parameters
2. Prepare the dataset
3. Get the model and setup the Trainer
4. Fine tuning and evaluation
5. Export the model
6. Reload the model and make predictions

## Running the notebook

The instructions below explain how to run the notebook on [bare metal](#bare-metal) using a
virtual environment.

1. Get a clone of the Model Zoo repository from GitHub:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Create a Python3.9 virtual environment and install the dependencies using the `requirements.txt` file.
   
   You can use virtualenv:
   ```
   python3.9 -m venv intel-pyt-nlp-venv
   source intel-pyt-nlp-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n intel-pyt-nlp python=3.9
   conda activate intel-pyt-nlp
   ```
   Then, from inside the activated virtualenv or conda environment run these steps:
   ```
   pip install --upgrade pip
   pip install -r ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/pytorch_text_classification/requirements.txt
   ```
3. Set environment variables for the path to the cache folder (for storing pretrained
   models and datasets) and an output directory. Both of these directories can be empty. Subsequent
   runs can reuse the same cache directory.
   ```
   export CACHE_DIR=<cache directory for models and datasets>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $CACHE_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/pytorch_text_classification
   jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run
   the notebook.
   
Dataset Citation:
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

@misc{misc_sms_spam_collection_228,
  author       = {Almeida, Tiago},
  title        = {{SMS Spam Collection}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository}
}
```

