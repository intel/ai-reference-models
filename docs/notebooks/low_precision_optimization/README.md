# Performance Optimization by Intel(R) Low Precision Optimization Tool (LPOT)

## Background
Performance of inference would be improved by converting fp32 model to int8 or bf16 model. Low-precision inference performance can be further improved by using AVX2, AVX512 and Intel(R) Deep Learning Boost technology supported in the Second Generation Intel(R) Xeon(R) Scalable Processors or later.

Intel(R) Low Precision Optimization Tool (LPOT) helps users to convert fp32 model to int8/bf16 easily and LPOT has less accuracy loss by the fine-tuned quantization method.

LPOT is a part of the Intel(R) AI Analytics Toolkit release.

Please refer to official website for detailed info and news: [https://github.com/intel/lp-opt-tool](https://github.com/intel/lp-opt-tool)

## License

This code sample is licensed under Apache License 2.0. See
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE) for details.


## Purpose

Demostrate how to quantize a pre-trained model from Model Zoo for Intel(R) Architecture by LPOT and show the performance gain after quantization.

## Supported AI Frameworks

LPOT supports Tensorflow*, Pytorch*, MXNet*, ONNX RT*.

| AI Framework | Sample| Description
| --------- | ------------------------------------------------ | -
| Tensorflow | [tensorflow](tensorflow) | quantize a pre-trained model from Intel(R) Model Zoo with LPOT


