# Question Answering fine tuning using TensorFlow

This notebook demonstrates fine tuning using various [BERT](https://arxiv.org/abs/1810.04805) models
from [TF Hub](https://tfhub.dev) using Intel® Optimization for TensorFlow with the SQuAD dataset.

The notebook performs the following steps:
1. Import dependencies and setup parameters
1. Prepare the dataset
1. Fine tuning and evaluation
1. Export the saved model

## Running the notebooks

To run the notebook, follow the instructions in `setup.md`.

## References

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
