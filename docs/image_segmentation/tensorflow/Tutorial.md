# Image Segmentation with 3D U-Net

## Goal
Learn CPU performance optimizations for image segmentation with [3D U-Net](https://arxiv.org/pdf/1606.06650.pdf) by optimizing TensorFlow to run faster on CPU.

## Abstract  
You can leverage [Intel®-optimized TensorFlow](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide) to improve inference performance of 3D U-Net (volumetric segmentation in medical imaging) on Intel CPUs. 

Intel®’s primitives library is called Intel® Math Kernel Library for Deep Neural Networks (MKL-DNN) and includes convolution, normalization, activation and inner product, and other primitives. 
These steps are highly relevant as recent academic articles predict the development of non-static neural networks that increase memory and computational requirements, especially where accuracy minimization is paramount.

KEYWORDS. Convolutional Neural Networks, Biomedical Volumetric Image Segmentation, TensorFlow Optimization.

## Background
[Image Segmentation](https://en.wikipedia.org/wiki/Image_segmentation) groups similar pixels of an image into different categories. 
It is often used to detect objects within an image and determine their boundaries. 
For example, image segmentation can be found in models that detect buildings in [satellite imagery](https://arxiv.org/pdf/1805.06561.pdf), pedestrians on [city streets](https://www.cityscapes-dataset.com/), and tumors in [medical images](https://arxiv.org/pdf/1705.03820.pdf). 

3D U-Net architecture consists of a contracting encoder section that analyses images to capture context and an expanding decoder section, producing full-resolution segmentations for precise localizations. 
It is end-to-end and trainable from few images, and consumes 3D volumes as inputs and processes them with 3D convolutions, max pooling, and upsampling.

For more background information:
  - [Original U-Net article](https://arxiv.org/pdf/1505.04597.pdf)
  - [3D U-Net article](https://arxiv.org/pdf/1606.06650.pdf)
  - [3D U-Net GitHub repository](https://github.com/ellisdg/3DUnetCNN)

## Recommended Settings 

The optimizations offered by Intel MKL-DNN allow TensorFlow to take advantage of the [advanced vector instruction sets](https://www.intel.ai/tensorflow-optimizations-intel-xeon-scalable-processor/) on modern Intel CPUs.
Additionally, there are several runtime settings which can also improvement inference speed.
Tuning these options to optimize CPU workloads is vital to enable max performance out of Intel-optimized TensorFlow. 
Below are the set of run-time options tested empirically on 3D U-Net and recommended by Intel.

| Run-time options  | Recommendations|
| ------------- | ------------- |
| Batch Size | 1 |
| Hyperthreading  | Enabled. Turn on in BIOS. Requires a restart. |
|intra_op_parallelism_threads| physical cores | 
|inter_op_parallelism_threads | 2 |
|Data Layout| NCHW|
|NUMA Controls| numactl --cpunodebind=0  --membind=0 |
|KMP_AFFINITY| granularity=fine,verbose,compact,1,0|
|KMP_BLOCKTIME| 1 |
|OMP_NUM_THREADS |physical cores|
 
*Note: Refer to [this article](https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference) to learn  more about the run time options.*

Run the following command to get the number of physical cores for your processor:
```
lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l
```

Below is a code snippet you can incorporate into your existing TensorFlow application to set the best settings. 
You can either set them in the CLI or in the Python script. Note that inter and intra_op_parallelism_threads settings can only be set in the Python script.

```bash
export OMP_NUM_THREADS=<# physical cores>
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
```
(or)
```
import os
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,compact,1,0"
os.environ["OMP_NUM_THREADS"]= <# physical cores>
config = tf.ConfigProto()
config.intra_op_parallelism_threads = <# physical cores>
config.inter_op_parallelism_threads = 2
tf.Session(config=config)
```

## Hands-on Tutorial
This section shows how to measure inference performance on Intel's Model Zoo pretrained 3D U-Net model by setting the above-discussed run time flags. 

For steps 1-3, refer to the Intel Model Zoo [3D U-Net FP32 benchmarks README](/benchmarks/image_segmentation/tensorflow/3d_unet#fp32-inference-instructions).

### Initial Setup
1. **Download and preprocess the BraTS dataset**: For a detailed walkthrough, see [here](/docs/image_segmentation/tensorflow/BraTS.md).
2. **Clone the intelai/models repo**:
   ```
   $ git clone https://github.com/IntelAI/models.git
   ```
3. **Download the pretrained model**: We are using the "Original U-Net" model from the [3DUnetCNN](https://github.com/ellisdg/3DUnetCNN/blob/master/README.md#pre-trained-models) repository.

4. **Install [Docker](https://docs.docker.com/v17.09/engine/installation/)**: This tutorial runs on a Docker container.

### Run Inference
1. Pull the relevant Intel-optimized TensorFlow Docker image. We'll be running the pretrained model to infer on Docker container.
   Click here to find all the available Docker images.
   ```
   docker pull docker.io/intelaipg/intel-optimized-tensorflow:latest-devel-mkl-py3
   ```
2. cd to the inference script directory.
   ```
   cd ~/models/benchmarks
   ```
3. Run the Python script `launch_benchmark.py` with the pretrained model and BraTS dataset. 
   The `launch_benchmark.py` script can be treated as an entry point to conveniently perform out-of-box high performance inference on pretrained models trained of popular topologies. 
   The script will automatically set the recommended run-time options for supported topologies, but if you choose to set your own options, refer to full of available flags and a detailed explanation on `launch_benchmark.py` script here. 
   This step will automatically launch a new container on every run and terminate. Go to the Step 4 to interactively run the script on the container.

   **Real-time Inference** and **Maximum Throughput Inference** (batch_size=1 for both)
   
    ```
    python launch_benchmark.py \
        --in-graph /home/<user>/tumor_segmentation_model.h5 \
        --data-location /home/<user>/3DUnetCNN/brats \
        --model-name 3d_unet \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --batch-size 1 \
        --benchmark-only \
        --socket-id 0 \
        --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl-py3
    ```

## Conclusion
The optimization of TensorFlow allows for deep learning models built for this common framework to run faster on Intel® processors to increase scaling and analytical flexibility. 
The Xeon® processor is designed to scale effectively to reduce training time of machine learning models. 
The collaboration between Intel® and Google engineers to optimize TensorFlow for higher performance on CPUs is part of ongoing efforts to increase the flexibility of AI applications by running on multiple running on multiple hardware platforms. 
Intel® believes the expansion of this accessibility is critical in the development of the next generation of AI models. 





