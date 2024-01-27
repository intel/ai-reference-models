# Text Classifier fine tuning with PyTorch

This notebook demonstrates fine tuning [pretrained models from Hugging Face](https://huggingface.co/models)
using text classification datasets from the [Hugging Face Datasets catalog](https://huggingface.co/datasets) or
a custom dataset. The [IMDb Larget Movie Review dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is used
from the Hugging Face Datasets catalog, and the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
is used as an example of a custom dataset being loaded from a csv file.

The notebook uses
[Intel® Extension for PyTorch\*](https://intel.github.io/intel-extension-for-pytorch) which extends PyTorch
with optimizations for extra performance boost on Intel hardware.

The notebook performs the following steps:
1. Import dependencies and setup parameters
2. Prepare the dataset
3. Prepare the Model for Fine Tuning and Evaluation
4. Export the model
5. Reload the model and make predictions

## Running the notebook

To run the notebook, follow the instructions to install the `setup.md`.

## References

Dataset Citations
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
Please see this dataset's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.

