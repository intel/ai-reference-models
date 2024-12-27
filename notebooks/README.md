# Run Intel® AI Reference Models in a Jupyter Notebook

This Jupyter notebook helps you choose and run a comparison between two models from the [Intel® AI Reference Models repo](https://github.com/IntelAI/models) using Intel® Optimizations for TensorFlow*. When you run the notebook, it installs required package dependencies, displays information about your platform, lets you choose the two models to compare, runs those models, and finally displays a performance comparison chart.

## Supported Models

#### Intel® Data Center CPU Workloads

| Model | Framework | Mode |  Supported Precisions |
| ----- | --------- | ---- | ----------- |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet)  [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | TensorFlow | Inference | [Int8 FP32 BFloat16 BFloat32](/quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/README.md) |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet)  [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | TensorFlow | Training |  [FP32 BFloat16 BFloat32](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/README.md) |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Inference | [Int8 FP32 BFloat16 BFloat32](/../models_v2/pytorch/resnet50/inference/cpu/README.md) |
| [ResNet 50](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Training  | [FP32 BFloat16 BFloat32](/../models_v2/pytorch/resnet50/training/cpu/README.md) |
| [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224) | PyTorch | Inference | [FP32 BFloat16 BFloat32 FP16 INT8](/../models_v2/pytorch/vit/inference/cpu/README.md) |
| [3D U-Net](https://arxiv.org/pdf/1606.06650.pdf) | TensorFlow | Inference | [FP32 BFloat16 Int8](/benchmarks/image_segmentation/tensorflow/3d_unet/inference/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | Tensorflow | Inference | [FP32 BFloat16 Int8 BFloat32](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) [Sapphire Rapids](https://www.intel.com/content/www/us/en/newsroom/opinion/updates-next-gen-data-center-platform-sapphire-rapids.html#gs.blowcx) | Tensorflow | Training | [FP32 BFloat16 BFloat32](/quickstart/language_modeling/tensorflow/bert_large/training/cpu/README.md) |
| [BERT large (Hugging Face)](https://arxiv.org/pdf/1810.04805.pdf) | TensorFlow | Inference | [FP32 FP16 BFloat16 BFloat32](/benchmarks/language_modeling/tensorflow/bert_large_hf/inference/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Inference | [FP32 Int8 BFloat16 BFloat32](/../models_v2/pytorch/bert_large/inference/cpu/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Training  | [FP32 BFloat16 BFloat32](/../models_v2/pytorch/bert_large/training/cpu/README.md) |
| [DistilBERT base](https://arxiv.org/abs/1910.01108)  | PyTorch | Inference | [FP32 BF32 BF16Int8-FP32 Int8-BFloat16 BFloat32](/../models_v2/pytorch/distilbert/inference/cpu/README.md) |
| [RNN-T](https://arxiv.org/abs/2007.15188)            | PyTorch | Inference | [FP32 BFloat16 BFloat32](/../models_v2/pytorch/rnnt/inference/cpu/README.md) |
| [RNN-T](https://arxiv.org/abs/2007.15188)            | PyTorch | Training  | [FP32 BFloat16 BFloat32](/../models_v2/pytorch/rnnt/training/cpu/README.md) |
| [GPTJ 6B](https://huggingface.co/EleutherAI/gpt-j-6b) | PyTorch | Inference | [FP32 FP16 BFloat16 BF32 INT8](/../models_v2/pytorch/gptj/inference/cpu/README.md) | |
| [GPTJ 6B MLPerf](https://github.com/mlcommons/inference/tree/master/language/gpt-j#datasets--models) | PyTorch | Inference | [INT4](/../models_v2/pytorch/gpt-j_mlperf/inference/cpu/README.md) |
| [ChatGLMv3 6B](https://huggingface.co/THUDM/chatglm3-6b) | PyTorch | Inference | [FP32 FP16 BFloat16 BF32 INT8](/../models_v2/pytorch/chatglm/inference/cpu/README.md) |
| [BERT](https://arxiv.org/pdf/1810.04805.pdf)                    | TensorFlow | Inference | [FP32](/benchmarks/language_translation/tensorflow/bert/inference/README.md) |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870)                | PyTorch | Inference  | [FP32 BFloat16 BFloat32](/../models_v2/pytorch/maskrcnn/inference/cpu/README.md) |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870)                | PyTorch | Training   | [FP32 BFloat16 BFloat32](/../models_v2/pytorch/maskrcnn/training/cpu/README.md) |
| [SSD-ResNet34](https://arxiv.org/abs/1512.02325)              | PyTorch | Inference  | [FP32 Int8 BFloat16 BFloat32](/../models_v2/pytorch/ssd-resnet34/inference/cpu/README.md) |
| [SSD-ResNet34](https://arxiv.org/abs/1512.02325)              | PyTorch | Training   | [FP32 BFloat16 BFloat32](/../models_v2/pytorch/ssd-resnet34/training/cpu/README.md) |
| [Yolo V7](https://arxiv.org/abs/2207.02696)              | PyTorch | Inference   | [Int8 FP32 FP16 BFloat16 BFloat32](/../models_v2/pytorch/yolov7/inference/cpu/README.md) |
| [Wide & Deep](https://arxiv.org/pdf/1606.07792.pdf) | TensorFlow | Inference | [FP32](/../models_v2/tensorflow/wide_deep/inference/README.md) |
| [DLRM](https://arxiv.org/pdf/1906.00091.pdf)         | PyTorch | Inference | [FP32 Int8 BFloat16 BFloat32](/../models_v2/pytorch/dlrm/inference/cpu/README.md) |
| [DLRM](https://arxiv.org/pdf/1906.00091.pdf)         | PyTorch | Training  | [FP32 BFloat16 BFloat32](/../models_v2/pytorch/dlrm/training/cpu/README.md) |
| [DLRM v2](https://arxiv.org/pdf/1906.00091.pdf)         | PyTorch | Inference | [FP32 FP16 BFloat16 BFloat32 Int8](/../models_v2/pytorch/torchrec_dlrm/inference/cpu/README.md) |
| [Stable Diffusion](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/) | TensorFlow | Inference | [FP32 BFloat16 FP16](../models_v2//tensorflow/stable_diffusion/inference/README.md) |
| [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1) | PyTorch | Inference | [FP32 BFloat16 FP16 BFloat32 Int8-FP32 Int8-BFloat16](/../models_v2/pytorch/stable_diffusion/inference/cpu/README.md) |
| [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1) | PyTorch | Training | [FP32 BFloat16 FP16 BFloat32](/../models_v2/pytorch/stable_diffusion/training/cpu/README.md) |
| [Latent Consistency Models(LCM)](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) | PyTorch | Inference | [FP32 BFloat16 FP16 BFloat32 Int8-FP32 Int8-BFloat16](/../models_v2/pytorch/LCM/inference/cpu/README.md) |
| [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf) | TensorFlow | Inference | [FP32 BFloat16 FP16 Int8 BFloat32](/../models_v2//tensorflow/graphsage/inference/README.md) |

#### Intel® Data Center CPU Workloads

| Model | Framework | Mode |  Platform  | Supported Precisions |
| ----- | --------- | ---- | ----------- | --------------- |
| [ResNet 50v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet) | TensorFlow | Inference | Flex Series | [Float32 TF32 Float16 BFloat16 Int8](../models_v2/tensorflow/resnet50v1_5/inference/gpu/README.md) |
| [ResNet 50 v1.5](https://github.com/tensorflow/models/tree/v2.11.0/official/legacy/image_classification/resnet) | TensorFlow | Training | Max Series | [BFloat16 FP32](../models_v2/tensorflow/resnet50v1_5/training/gpu/README.md) |
| [ResNet 50 v1.5](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Inference | Flex Series, Max Series, Arc Series |[Int8 FP32 FP16 TF32](../models_v2/pytorch/resnet50v1_5/inference/gpu/README.md) |
| [ResNet 50 v1.5](https://arxiv.org/pdf/1512.03385.pdf)    | PyTorch | Training | Max Series, Arc Series |[BFloat16 TF32 FP32](../models_v2/pytorch/resnet50v1_5/training/gpu/README.md) |
| [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) | PyTorch | Inference | Flex Series, Max Series | [FP32 FP16 BF16 TF32](../models_v2/pytorch/distilbert/inference/gpu/README.md) |
| [DLRM v1](https://arxiv.org/pdf/1906.00091.pdf) | PyTorch | Inference | Flex Series | [FP16 FP32](../models_v2/pytorch/dlrm/inference/gpu/README.md) |
| [SSD-MobileNet*](https://arxiv.org/pdf/1704.04861.pdf)| PyTorch | Inference | Arc Series| [INT8 FP16 FP32](../models_v2/pytorch/ssd-mobilenetv1/inference/gpu/README.md) |
| [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)  | PyTorch | Inference | Flex Series | [FP16 BF16 FP32](../models_v2/pytorch/efficientnet/inference/gpu/README.md) |
| [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)  | TensorFlow | Inference | Flex Series | [FP16](../models_v2/tensorflow/efficientnet/inference/gpu/README.md) |
| [FBNet](https://arxiv.org/pdf/1812.03443.pdff)  | PyTorch | Inference | Flex Series | [FP16 BF16 FP32](../models_v2/pytorch/fbnet/inference/gpu/README.md) |
| [Wide Deep Large Dataset](https://arxiv.org/pdf/2112.10752.pdf)  | TensorFlow | Inference | Flex Series | [FP16](../models_v2/tensorflow/wide_deep_large_ds/inference/gpu/README.md) |
| [YOLO V5](https://arxiv.org/pdf/2108.11539.pdf)  | PyTorch | Inference | Flex Series | [FP16](../models_v2/pytorch/yolov5/inference/gpu/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Inference | Max Series, Arc Series | [BFloat16 FP32 FP16](../models_v2/pytorch/bert_large/inference/gpu/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf)   | PyTorch | Training  | Max Series, Arc Series | [BFloat16 FP32 TF32](../models_v2/pytorch/bert_large/training/gpu/README.md) |
| [BERT large](https://arxiv.org/pdf/1810.04805.pdf) | TensorFlow | Training | Max Series | [BFloat16 TF32 FP32](../models_v2/tensorflow/bert_large/training/gpu/README.md) |
| [DLRM v2](https://arxiv.org/abs/1906.00091) | PyTorch | Inference | Max Series | [FP32 BF16](../models_v2/pytorch/torchrec_dlrm/inference/gpu/README.md)
| [DLRM v2](https://arxiv.org/abs/1906.00091) | PyTorch | Training | Max Series | [FP32 TF32 BF16](../models_v2/pytorch/torchrec_dlrm/training/gpu/README.md)
| [3D-Unet](https://arxiv.org/pdf/1606.06650.pdf) | PyTorch | Inference | Max Series | [FP16 INT8 FP32](../models_v2/pytorch/3d_unet/inference/gpu/README.md) |
| [3D-Unet](https://arxiv.org/pdf/1606.06650.pdf) | TensorFlow | Training | Max Series | [BFloat16 FP32](../models_v2/tensorflow/3d_unet/training/gpu/README.md) |
| [Stable Diffusion](https://arxiv.org/pdf/2112.10752.pdf)  | PyTorch | Inference | Flex Series, Max Series, Arc Series | [FP16 FP32](../models_v2/pytorch/stable_diffusion/inference/gpu/README.md) |
| [Stable Diffusion](https://arxiv.org/pdf/2112.10752.pdf)  | TensorFlow | Inference | Flex Series | [FP16 FP32](../models_v2/tensorflow/stable_diffusion/inference/gpu/README.md) |
| [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)  | TensorFlow | Inference | Flex Series | [FP32 Float16](../models_v2/tensorflow/maskrcnn/inference/gpu/README.md) |
| [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)  | TensorFlow | Training | Max Series | [FP32 BFloat16](../models_v2/tensorflow/maskrcnn/training/gpu/README.md) |
| [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf)  | PyTorch | Inference | Flex Series | [FP16](../models_v2/pytorch/swin-transformer/inference/gpu/README.md) |
| [FastPitch](https://arxiv.org/pdf/1703.06870.pdf)  | PyTorch | Inference | Flex Series | [FP16](../models_v2/pytorch/fastpitch/inference/gpu/README.md) |
| [UNet++](https://arxiv.org/pdf/1807.10165.pdf)  | PyTorch | Inference | Flex Series | [FP16](../models_v2/pytorch/unetpp/inference/gpu/README.md) |
| [RNN-T](https://arxiv.org/abs/1211.3711) | PyTorch | Inference | Max Series | [FP16 BF16 FP32](../models_v2/pytorch/rnnt/inference/gpu/README.md) |
| [RNN-T](https://arxiv.org/abs/1211.3711) | PyTorch | Training | Max Series | [FP32 BF16 TF32](../models_v2/pytorch/rnnt/training/gpu/README.md) |
| [IFRNet](https://arxiv.org/pdf/2205.14620.pdf)  | PyTorch | Inference | Flex Series | [FP16](../models_v2/pytorch/IFRNet/inference/gpu/README.md) |
| [RIFE](https://arxiv.org/pdf/2011.06294.pdf)    | PyTorch | Inference | Flex Series | [FP16](../models_v2/pytorch/RIFE/inference/gpu/README.md) |

## Environment Setup

Instead of installing or updating packages system-wide, it's a good idea to install project-specific Python packages in a Python virtual environment localized to your project. The Python virtualenv package lets you do just that.  Using virtualenv is optional, but recommended.

The jupyter notebook runs on Ubuntu distribution for Linux.

 1. **Virtualenv Python Environment**
       Install virtualenv on Ubuntu using these commands:
       ```
       sudo apt-get update
       sudo apt-get install python-dev python-pip
       sudo pip install -U virtualenv  # system-wide install
       ```

       Activate virtual environment using the following command:
       ```
       virtualenv -p python ai_ref_models
       source ai_ref_models/bin/activate
       ```

 2. **Jupyter Notebook Support**:

       Install Jupyter notebook support with the command:
       ```
          pip install notebook
       ```
       Refer to the [Installing Jupyter guide](https://jupyter.org/install) for details.


## How to Run the Notebook

1. Clone the Intel® AI Reference Models repo:
   ```
   git clone https://github.com/IntelAI/models.git
   ```
2. Launch the Jupyter notebook server: `jupyter notebook --ip=0.0.0.0`
3. Follow the instructions to open the URL with the token in your browser, something like this: ` http://127.0.0.1:8888/tree?token=<token>`
4. Browse to the `models/notebooks/` folder
5. Click the AI_Reference_Models notebook file - [AI_Reference_Models.ipynb](https://github.com/IntelAI/models/notebooks/AI_Reference_Models.ipnyb).
6. Read the instructions and run through each notebook cell, in order, ending with a display of the analysis results. Note that some cells prompt you for input, such as selecting the model number you'd like to run.
7. When done, you should deactivate the virtualenv, if you used one, with the command: `deactivate`
