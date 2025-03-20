<a href="https://scan.coverity.com/projects/intel-models">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/30278/badge.svg"/>
</a>

</a>
<a href="https://www.bestpractices.dev/projects/8925"><img src="https://www.bestpractices.dev/projects/8925/badge"></a>

# Intel® AI Reference Models

This repository contains **links to pre-trained models, sample scripts, best practices, and step-by-step tutorials** for many popular open-source machine learning models optimized by Intel to run on Intel® Xeon® Scalable processors and Intel® Data Center GPUs.

Containers for running the workloads can be found at [Intel® AI Containers](https://github.com/intel/ai-containers/blob/v0.3.5/docs/README.md).

[Intel® AI Reference Models in a Jupyter Notebook](/notebooks/README.md) is also available for the [listed workloads](/notebooks/README.md#supported-models)

## Purpose of Intel® AI Reference Models

Intel optimizes popular deep learning frameworks such as TensorFlow* and PyTorch* by contributing to the upstream projects. Additional optimizations are built into plugins/extensions such as the [Intel Extension for Pytorch*](https://github.com/intel/intel-extension-for-pytorch) and the [Intel Extension for TensorFlow*](https://github.com/intel/intel-extension-for-tensorflow). Popular neural network models running against common datasets are the target workloads that drive these optimizations.

The purpose of the Intel® AI Reference Models repository (and associated containers) is to quickly replicate the complete software environment that demonstrates the best-known performance of each of these target model/dataset combinations. When executed in optimally-configured hardware environments, these software environments showcase the AI capabilities of Intel platforms.

***DISCLAIMER: These scripts are not intended for benchmarking Intel platforms.
For any performance and/or benchmarking information on specific Intel platforms, visit [https://www.intel.ai/blog](https://www.intel.ai/blog).***

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.

## License
The Intel® AI Reference Models is licensed under [Apache License Version 2.0](https://github.com/intel/ai-reference-models/blob/master/LICENSE).

## Datasets
To the extent that any public datasets are referenced by Intel or accessed using tools or code on this site those datasets are provided by the third party indicated as the data source. Intel does not create the data, or datasets, and does not warrant their accuracy or quality. By accessing the public dataset(s) you agree to the terms associated with those datasets and that your use complies with the applicable license.

Please check the list of datasets used in Intel® AI Reference Models in [datasets directory](/datasets).

Intel expressly disclaims the accuracy, adequacy, or completeness of any public datasets, and is not liable for any errors, omissions, or defects in the data, or for any reliance on the data. Intel is not liable for any liability or damages relating to your use of public datasets.

## Use cases
The model documentation in the tables below have information on the
prerequisites to run each model. The model scripts run on Linux. Certain
models are also able to run using bare metal on Windows. For more information
and a list of models that are supported on Windows, see the
[documentation here](/docs/general/Windows.md#using-intel-ai-reference-models-on-windows-systems).

Instructions available to run on [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx).

For best performance on Intel® Data Center GPU Max Series, please check the [list of supported workloads](#intel-data-center-gpu-workloads). It provides instructions to run inference and training using [Intel(R) Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) or [Intel(R) Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow).

### Image Recognition

| Model                                                  | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ------------------------------------------------------ | ---------- | ----------| ------------------- | ---------------------- |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet)  [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | TensorFlow | Inference | [Int8 FP32 BFloat16 BFloat32](/models_v2/tensorflow/resnet50v1_5/inference/cpu/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet)  [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | TensorFlow | Training |  [FP32 BFloat16 BFloat32](/models_v2/tensorflow/resnet50v1_5/training/cpu/README.md) | [ImageNet 2012](https://github.com/IntelAI/models/tree/master/datasets/imagenet/README.md) |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Inference | [Int8 FP32 BFloat16 BFloat32](/models_v2/pytorch/resnet50/inference/cpu/README.md) | [ImageNet 2012] |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Training  | [FP32 BFloat16 BFloat32](/models_v2/pytorch/resnet50/training/cpu/README.md) | [ImageNet 2012] |
| [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224) | PyTorch | Inference | [FP32 BFloat16 BFloat32 FP16 INT8](/models_v2/pytorch/vit/inference/cpu/README.md) | [ImageNet 2012] |

### Image Segmentation

| Model                                                    | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| -------------------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [3D U-Net](https://arxiv.org/pdf/1606.06650.pdf) | TensorFlow | Inference | [FP32 BFloat16 Int8](/models_v2/tensorflow/3d_unet/inference/cpu/README.md) | [BRATS 2018](https://www.med.upenn.edu/sbia/brats2018/registration.html) |

### Language Modeling

| Model                                        | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| -------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | Tensorflow | Inference | [FP32 BFloat16 Int8 BFloat32](/models_v2/tensorflow/bert_large/inference/cpu/README.md) | [SQuAD](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#inference) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | Tensorflow | Training | [FP32 BFloat16 BFloat32](/models_v2/tensorflow/bert_large/training/cpu/README.md) | [SQuAD](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#inference) |
| [BERT large (Hugging Face)](https://arxiv.org/pdf/1810.04805.pdf) | TensorFlow | Inference | [FP32 FP16 BFloat16 BFloat32](/models_v2/tensorflow/bert_large_hf/inference/cpu/README.md) | [SQuAD](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#inference) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Inference | [FP32 Int8 BFloat16 BFloat32](/models_v2/pytorch/bert_large/inference/cpu/README.md) | BERT Large SQuAD1.1 |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Training  | [FP32 BFloat16 BFloat32](/models_v2/pytorch/bert_large/training/cpu/README.md) | [preprocessed text dataset](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v) |
| [DistilBERT base](https://arxiv.org/abs/1910.01108)  | PyTorch | Inference | [FP32 BF32 BF16Int8-FP32 Int8-BFloat16 BFloat32](/models_v2/pytorch/distilbert/inference/cpu/README.md) | [ DistilBERT Base SQuAD1.1](https://huggingface.co/distilbert-base-uncased-distilled-squad) |
| [RNN-T](https://arxiv.org/abs/2007.15188)            | PyTorch | Inference | [FP32 BFloat16 BFloat32](/models_v2/pytorch/rnnt/inference/cpu/README.md) | [RNN-T dataset](/models_v2/pytorch/rnnt/inference/cpu/download_dataset.sh) |
| [RNN-T](https://arxiv.org/abs/2007.15188)            | PyTorch | Training  | [FP32 BFloat16 BFloat32](/models_v2/pytorch/rnnt/training/cpu/README.md) | [RNN-T dataset](/models_v2/pytorch/rnnt/training/cpu/download_dataset.sh) |
| [GPTJ 6B](https://huggingface.co/EleutherAI/gpt-j-6b) | PyTorch | Inference | [FP32 FP16 BFloat16 BF32 INT8](/models_v2/pytorch/gptj/inference/cpu/README.md) | |
| [LLAMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) | PyTorch | Inference | [FP32 FP16 BFloat16 BF32 INT8](/models_v2/pytorch/llama/inference/cpu/README.md) | |
| [LLAMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) | PyTorch | Training | [FP32 FP16 BFloat16 BF32](/models_v2/pytorch/llama/training/cpu/README.md) | |
| [LLAMA2 13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) | PyTorch | Inference | [FP32 FP16 BFloat16 BF32 INT8](/models_v2/pytorch/llama/inference/cpu/README.md) | |
| [ChatGLMv3 6B](https://huggingface.co/THUDM/chatglm3-6b) | PyTorch | Inference | [FP32 FP16 BFloat16 BF32 INT8](/models_v2/pytorch/chatglm/inference/cpu/README.md) | |


### Language Translation

| Model                                                           | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| --------------------------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [BERT](https://arxiv.org/pdf/1810.04805.pdf)                    | TensorFlow | Inference | [FP32](/models_v2/tensorflow/bert/inference/cpu/README.md) | [MRPC](https://github.com/IntelAI/models/tree/master/datasets/bert_data/README.md#classification-training-with-bert) |

### Object Detection

| Model                                                 | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ----------------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870)                | PyTorch | Inference  | [FP32 BFloat16 BFloat32](/models_v2/pytorch/maskrcnn/inference/cpu/README.md) | [COCO 2017](/models_v2/pytorch/maskrcnn/inference/cpu/README.md#datasets) |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870)                | PyTorch | Training   | [FP32 BFloat16 BFloat32](/models_v2/pytorch/maskrcnn/training/cpu/README.md) | [COCO 2017](/models_v2/pytorch/maskrcnn/training/cpu/README.md#datasets) |
| [SSD-ResNet34](https://arxiv.org/abs/1512.02325)              | PyTorch | Inference  | [FP32 Int8 BFloat16 BFloat32](/models_v2/pytorch/ssd-resnet34/inference/cpu/README.md) | [COCO 2017](/models_v2/pytorch/ssd-resnet34/inference/cpu/README.md) |
| [SSD-ResNet34](https://arxiv.org/abs/1512.02325)              | PyTorch | Training   | [FP32 BFloat16 BFloat32](/models_v2/pytorch/ssd-resnet34/training/cpu/README.md) | [COCO 2017](/models_v2/pytorch/ssd-resnet34/training/cpu/README.md) |
| [Yolo V7](https://arxiv.org/abs/2207.02696)              | PyTorch | Inference   | [Int8 FP32 FP16 BFloat16 BFloat32](/models_v2/pytorch/yolov7/inference/cpu/README.md) | [COCO 2017](/models_v2/pytorch/yolov7/inference/cpu/README.md## Prepare Dataset) |

### Recommendation

| Model                                                  | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ------------------------------------------------------ | ---------- | ----------| ------------------- | ---------------------- |
| [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf) | TensorFlow | Inference | [FP32](/models_v2/tensorflow/wide_deep/inference/cpu/README.md) | [Census Income dataset](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep/inference/fp32#dataset) |
| [DLRM](https://arxiv.org/pdf/1906.00091.pdf)         | PyTorch | Inference | [FP32 Int8 BFloat16 BFloat32](/models_v2/pytorch/dlrm/inference/cpu/README.md) | [Criteo Terabyte](/models_v2/pytorch/dlrm/inference/cpu/README.md#datasets) |
| [DLRM](https://arxiv.org/pdf/1906.00091.pdf)         | PyTorch | Training  | [FP32 BFloat16 BFloat32](/models_v2/pytorch/dlrm/training/cpu/README.md) | [Criteo Terabyte](/models_v2/pytorch/dlrm/training/cpu/README.md#datasets) |
| [DLRM v2](https://arxiv.org/pdf/1906.00091.pdf)         | PyTorch | Inference | [FP32 FP16 BFloat16 BFloat32 Int8](/models_v2/pytorch/torchrec_dlrm/inference/cpu/README.md) | [Criteo 1TB Click Logs dataset](/models_v2/pytorch/torchrec_dlrm/inference/cpu#datasets) |

### Diffusion

| Model                                           | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ----------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [Stable Diffusion](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/) | TensorFlow | Inference | [FP32 BFloat16 FP16](/models_v2/tensorflow/stable_diffusion/inference/cpu/README.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images)
| [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1) | PyTorch | Inference | [FP32 BFloat16 FP16 BFloat32 Int8-FP32 Int8-BFloat16](/models_v2/pytorch/stable_diffusion/inference/cpu/README.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images)
| [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1) | PyTorch | Training | [FP32 BFloat16 FP16 BFloat32](/models_v2/pytorch/stable_diffusion/training/cpu/README.md) | [cat images](https://huggingface.co/datasets/diffusers/cat_toy_example)
| [Latent Consistency Models(LCM)](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) | PyTorch | Inference | [FP32 BFloat16 FP16 BFloat32 Int8-FP32 Int8-BFloat16](/models_v2/pytorch/LCM/inference/cpu/README.md) | [COCO 2017 validation dataset](https://github.com/IntelAI/models/tree/master/datasets/coco#download-and-preprocess-the-coco-validation-images)

### Graph Networks

| Model                                           | Framework  | Mode      | Model Documentation | Benchmark/Test Dataset |
| ----------------------------------------------- | ---------- | ----------| ------------------- | ---------------------- |
| [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf) | TensorFlow | Inference | [FP32 BFloat16 FP16 Int8 BFloat32](/models_v2/tensorflow/graphsage/inference/cpu/README.md) | [Protein Protein Interaction](http://snap.stanford.edu/graphsage) |

*Means the model belongs to [MLPerf](https://mlperf.org/) models and will be supported long-term.


## Intel® Data Center GPU Workloads
| Model                             | Framework  | Mode      | GPU Type | Model Documentation |
| ----------------------------------| ---------- | ----------| -------- | ------------------- |
| [ResNet 50 v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet) | TensorFlow | Training | Max Series | [BFloat16 FP32](/models_v2/tensorflow/resnet50v1_5/training/gpu/README.md) |
| [ResNet 50 v1.5](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Inference | Max Series, Arc Series |[Int8 FP32 FP16 TF32](/models_v2/pytorch/resnet50v1_5/inference/gpu/README.md) |
| [ResNet 50 v1.5](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Training | Max Series, Arc Series |[BFloat16 TF32 FP32](/models_v2/pytorch/resnet50v1_5/training/gpu/README.md) |
| [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) | PyTorch | Inference | Max Series | [FP32 FP16 BF16 TF32](/models_v2/pytorch/distilbert/inference/gpu/README.md) |
| [SSD-MobileNet*](https://arxiv.org/pdf/1704.04861.pdf)| PyTorch | Inference | Arc Series| [INT8 FP16 FP32](/models_v2/pytorch/ssd-mobilenetv1/inference/gpu/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Inference | Max Series, Arc Series | [BFloat16 FP32 FP16](/models_v2/pytorch/bert_large/inference/gpu/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Training  | Max Series, Arc Series | [BFloat16 FP32 TF32](/models_v2/pytorch/bert_large/training/gpu/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) | TensorFlow | Training | Max Series | [BFloat16 TF32 FP32](/models_v2/tensorflow/bert_large/training/gpu/README.md) |
| [DLRM v2](https://arxiv.org/abs/1906.00091) | PyTorch | Inference | Max Series | [FP32 BF16](/models_v2/pytorch/torchrec_dlrm/inference/gpu/README.md)
| [DLRM v2](https://arxiv.org/abs/1906.00091) | PyTorch | Training | Max Series | [FP32 TF32 BF16](/models_v2/pytorch/torchrec_dlrm/training/gpu/README.md)
| [3D-Unet](https://arxiv.org/pdf/1606.06650.pdf) | PyTorch | Inference | Max Series | [FP16 INT8 FP32](/models_v2/pytorch/3d_unet/inference/gpu/README.md) |
| [3D-Unet](https://arxiv.org/pdf/1606.06650.pdf) | TensorFlow | Training | Max Series | [BFloat16 FP32](/models_v2/tensorflow/3d_unet/training/gpu/README.md) |
| [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)  | TensorFlow | Training | Max Series | [FP32 BFloat16](/models_v2/tensorflow/maskrcnn/training/gpu/README.md) |
| [RNN-T](https://arxiv.org/abs/1211.3711) | PyTorch | Inference | Max Series | [FP16 BF16 FP32](/models_v2/pytorch/rnnt/inference/gpu/README.md) |
| [RNN-T](https://arxiv.org/abs/1211.3711) | PyTorch | Training | Max Series | [FP32 BF16 TF32](/models_v2/pytorch/rnnt/training/gpu/README.md) |

## How to Contribute
If you would like to add a new benchmarking script, please use [this guide](/CONTRIBUTING.md).
