# Text Classifier fine tuning with TensorFlow

These notebooks demonstrate fine tuning using various [BERT](https://arxiv.org/abs/1810.04805) models
from [TF Hub](https://tfhub.dev) using Intel® Optimization for TensorFlow for text classification.

The notebook performs the following steps:
1. Install dependencies and setup parameters
1. Prepare the dataset
1. Build the model
1. Fine tuning and evaluation
1. Export the model
1. Reload the model and make predictions

## Running the notebook

To run the notebook, follow the instructions in `setup.md`.

## References

Dataset citations:
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

@misc{zhang2015characterlevel,
    title={Character-level Convolutional Networks for Text Classification},
    author={Xiang Zhang and Junbo Zhao and Yann LeCun},
    year={2015},
    eprint={1509.01626},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

@misc{misc_sms_spam_collection_228,
  author       = {Almeida, Tiago},
  title        = {{SMS Spam Collection}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository}
}
```
Please see this dataset's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.