# Large language model instruction tuning with PyTorch

This notebook demonstrates instruction tuning [pretrained causal language models from Hugging Face](https://huggingface.co/models)
using text generation datasets from the [Hugging Face Datasets catalog](https://huggingface.co/datasets) or
a custom dataset. The [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset is used
from the Hugging Face Datasets catalog, and a subset of the [Code Alpaca](https://github.com/sahil280114/codealpaca) dataset
is used as an example of a custom dataset being loaded from a json file.

The notebook includes options for bfloat16 precision training,
[Intel® Extension for PyTorch\*](https://intel.github.io/intel-extension-for-pytorch) which extends PyTorch
with optimizations for extra performance boost on Intel hardware, and [SmoothQuant quantization with Intel® Neural Compressor](https://github.com/intel/neural-compressor/blob/v2.1.1/docs/source/smooth_quant.md).

The notebook performs the following steps:
1. Import dependencies and setup parameters
2. Prepare the dataset
3. Prepare the model and test domain knowledge
4. Transfer Learning
5. Retest domain knowledge
6. Quantize the model

## Running the notebook

To run the notebook, follow the instructions to install the `setup.md`.

## References

Dataset Citations

<b>[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)</b> - Copyright (2023) Databricks, Inc. This dataset was developed at Databricks (https://www.databricks.com) and its use is subject to the CC BY-SA 3.0 license. Certain categories of material in the dataset include materials from the following sources, licensed under the CC BY-SA 3.0 license: Wikipedia (various pages) - https://www.wikipedia.org/ Copyright © Wikipedia editors and contributors.

```
@software{together2023redpajama,
  author = {Together Computer},
  title = {RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset},
  month = April,
  year = 2023,
  url = {https://github.com/togethercomputer/RedPajama-Data}
}
```

```
@misc{codealpaca,
  author = {Sahil Chaudhary},
  title = {Code Alpaca: An Instruction-following LLaMA model for code generation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sahil280114/codealpaca}},
}
```
