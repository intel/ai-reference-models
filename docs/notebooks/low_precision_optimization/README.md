# Performance Optimization by Intel® Neural Compressor

## Background
Performance of inference would be improved by converting fp32 model to int8 or bf16 model. Low-precision inference performance can be further improved by using AVX2, AVX512 and Intel(R) Deep Learning Boost technology supported in the Second Generation Intel(R) Xeon(R) Scalable Processors or later.

Intel® Neural Compressor helps users to convert fp32 model to int8/bf16 easily and has less accuracy loss by the fine-tuned quantization method. At the same time, this tool optimizes the model by more methods, like Pruning, Graphic Optimization.

Intel® Neural Compressor is a part of the Intel(R) AI Analytics Toolkit release. It can be installed within Intel(R) AI Analytics Toolkit from PyPi or Conda. It can also be compiled from source code independently. Please refer to official website for detailed info and news: [https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor).

## License

This code sample is licensed under Apache License 2.0. See
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE) for details.


## Purpose

Demostrate how to quantize a pre-trained model from Model Zoo for Intel(R) Architecture by Intel® Neural Compressor and show the performance gain after quantization.

## Supported AI Frameworks

Intel® Neural Compressor supports Tensorflow*, Pytorch*, MXNet*, ONNX RT*.

| AI Framework | Sample| Description|
| --------- | ------------------------------------------------ | --|
| Tensorflow | [tensorflow](tensorflow) | quantize a pre-trained model from Intel(R) Model Zoo by Intel® Neural Compressor|



