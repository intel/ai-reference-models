# Object Detection with SSD-VGG16


## Goal
This tutorial will introduce CPU performance considerations of the deep learning SSD-VGG16 model for object detection and how to use Intel® Optimizations for TensorFlow to improve inference time on CPUs. 
This tutorial will also provide code examples to use Intel Model Zoo's pretrained SSD-VGG16 model on the COCO dataset that can be copy/pasted for a quick off-the-ground implementation on real data.

## Background

Object detection is breaking into a wide range of industrial applications with some of the top uses cases including computer vision, surveillance, automated vehicle system, etc. One of the widely used topologies used in this space is SSD-VGG16 for its popularity to speed-up real-time inference. Unlike Faster R-CNN, SSD-VGG16 eliminates the need of region-proposal-network to predict the boundary boxes but uses feature maps instead.  The modeling process is split into 2 parts.

 1. Feature extraction where the base network is a collection on VGG16 convolution layers and the output from this layer is fed into the detection phase.<br>
 2. Detection where the entire network is a sequence of CNNs progressively decreasing in size extracting features and reducing the feature maps.  Each added feature layer in the CNNs produces a fixed set of detection predictions with a fixed-size collection of bounding boxes and scores for the presence of class instances in those boxes, followed by a non-maximum suppression step to produce the final detections.<br>

##  Recommended Settings 
In addition to TensorFlow optimizations that use the Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) to utilize instruction sets appropriately, the runtime settings also significantly contribute to improved performance. 
Tuning these options to optimize CPU workloads is vital to optimize the performance of TensorFlow on Intel processors. 
Below are the set of run-time options tested empirically and recommended by Intel for two Intel® Xeon scalable processors on an optimized SSD-VGG16 pretrained model.

<table class="tg">
  <tr>
    <th class="tg-amwm" rowspan="2">Run-time options</th>
    <th class="tg-amwm" colspan="2">Recommendations for Intel® Xeon Scalable Processors</th>
  </tr>
  <tr>
    <td class="tg-amwm">28-core 1st gen</td>
    <td class="tg-amwm">28-core 2nd gen</td>
  </tr>
  <tr>
    <td class="tg-0lax">Hyperthreading</td>
    <td class="tg-baqh" colspan="2">Enabled. Turn on in BIOS. Requires a restart.</td>
  </tr>
  <tr>
    <td class="tg-0lax">intra_op_parallelism_threads</td>
    <td class="tg-baqh">14</td>
    <td class="tg-baqh">21</td>
  </tr>
  <tr>
    <td class="tg-0lax">inter_op_parallelism_threads</td>
    <td class="tg-baqh">2</td>
    <td class="tg-baqh">11</td>
  </tr>
    <tr>
    <td class="tg-0lax">data_num_inter_threads</td>
    <td class="tg-baqh">2</td>
    <td class="tg-baqh">21</td>
  </tr>
  <tr>
    <td class="tg-0lax">data_num_intra_threads</td>
    <td class="tg-baqh">7</td>
    <td class="tg-baqh">28</td>
  </tr>
  <tr>
    <td class="tg-0lax">NUMA Controls</td>
    <td class="tg-baqh" colspan="2">--cpunodebind=0 --membind=0</td>
  </tr>
  <tr>
    <td class="tg-0lax">KMP_AFFINITY</td>
    <td class="tg-baqh" colspan="2">KMP_AFFINITY=granularity=fine,verbose,compact,1,0</td>
  </tr>
  <tr>
    <td class="tg-0lax">KMP_BLOCKTIME</td>
    <td class="tg-baqh" colspan="2">1</td>
  </tr>
  <tr>
    <td class="tg-0lax">OMP_NUM_THREADS</td>
    <td class="tg-baqh" colspan="2">28</td>
  </tr>
</table>
 
Note 1: Refer to this [link](https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference) to learn more about the run time options.


Below is a code snippet you can incorporate into your existing TensorFlow application to set the best settings. The values shown are for a 2nd gen Xeon scalable processor.
Note that these recommended settings are hardware and dataset-specific (COCO dataset). These settings are provided to give users a good starting point to tune model run-time settings and may vary based on the hardware choice.
You can either set them in the CLI or the Python script. Note that inter and intra_op_parallelism_threads settings can only be set 
in the Python script.

```bash
export OMP_NUM_THREADS=11
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
```
(or)
```
import os
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"]= 11
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 21
config.inter_op_parallelism_threads = 11
inference_sess = tf.Session(config=config)
```
The data config settings are used to parallelize the part of the graph concerned with image processing (the data layer).
```
data_config = tf.ConfigProto()
data_config.inter_op_parallelism_threads = 21
data_config.intra_op_parallelism_threads = 28
data_sess = tf.Session(config=data_config)
```

## Hands-on Tutorial
This section shows how to measure inference performance on Intel's Model Zoo pretrained model (or your pretrained model) by setting the above-discussed run time flags. 
The setting values shown in this tutorial are for a 2nd gen Xeon scalable processor.
### FP32 inference
 
### Initial Setup



1. Clone the original model repo and checkout the appropriate commit. For demonstration purpose, this entire tutorial will be implemented on the home directory, modify this location as required.

```
cd ~
mkdir object_detection
cd object_detection
mkdir ssd_vgg16
cd ssd_vgg16
```

```
git clone https://github.com/HiKapok/SSD.TensorFlow.git
cd SSD.TensorFlow
git checkout 2d8b0cb9b2e70281bf9dce438ff17ffa5e59075c
```

2. Clone IntelAI models and download into your home directory or pull the latest version.
```bash
cd ~
git clone https://github.com/IntelAI/models.git
```
(or)

```
cd ~/models
git pull
```

3. Skip to step 4 if you already have a pretrained model, or download the pretrained model from Intel Model Zoo. Find more info in the [README](/benchmarks/object_detection/tensorflow/ssd_vgg16#fp32-inference-instructions) doc.

```
cd ~/object_detection/ssd_vgg16
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssdvgg16_fp32_pretrained_model.pb
```
4. Skip to step 5 if you already have a dataset with annotations in TFRecords format or follow the below instructions to download and convert COCO dataset with annotations to TFRecords.
Note that to compute accuracy, the TFRecord's filename pattern must start with `"val-*"`

Download validation dataset:

```
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```
Download annotations for COCO dataset:

```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
unzip annotations_trainval2017.zip
```
Generate TFRecords by copying the `generate_coco_records.py` script available in IntelAI `models` repo into original model repo:

```
cp ~/models/models/object_detection/tensorflow/ssd_vgg16/inference/generate_coco_records.py ~/object_detection/ssd_vgg16/SSD.TensorFlow/dataset
```

Create an output directory to save the generated TFRecords:

```
mkdir ~/object_detection/ssd_vgg16/data_tfrecords
```

Some dependencies are required to run the script such as python3, TensorFlow and tqdm.
You can use the following install commands to install the requirements:

```
sudo apt-get install python3-venv python3-dev
pip3 install --upgrade pip
python3 -m venv venv
source venv/bin/activate
pip3 install intel-tensorflow tqdm
```
Run the dataset conversion script to generate TFRecords:

```
cd ~/object_detection/ssd_vgg16/SSD.TensorFlow/dataset
python generate_coco_records.py \
--image_path ~/object_detection/ssd_vgg16/val2017/ \
--annotations_file ~/object_detection/ssd_vgg16/annotations/instances_val2017.json \
--output_prefix val \
--output_path ~/object_detection/ssd_vgg16/data_tfrecords
```
The generated TFrecords can be found at the `--output_path`.

```
$ ls  ~/object_detection/ssd_vgg16/data_tfrecords

val-00000-of-00005  val-00001-of-00005  val-00002-of-00005  val-00003-of-00005  val-00004-of-00005

```

5. Install [Docker](https://docs.docker.com/v17.09/engine/installation/) since the tutorial runs in a Docker container.

### Run online inference

1. Pull the relevant Intel-optimized TensorFlow Docker image.
```bash
docker pull gcr.io/deeplearning-platform-release/tf-cpu.1-14
```
2. cd to the inference script directory in local IntelAI models repo.
```bash        
cd ~/models/benchmarks
```
3. Run the Python script ``` launch_benchmark.py``` with the pretrained model. 
The ```launch_benchmark.py``` script can be treated as an entry point to conveniently perform out-of-box high performance 
inference on pretrained models for popular topologies. 
The script will automatically set the recommended run-time options for supported topologies, 
but if you choose to set your own options, refer to the full list of available flags and a detailed
explanation of ```launch_benchmark.py``` [here](/docs/general/tensorflow/LaunchBenchmark.md).
 This step will automatically launch a new container on every run and terminate. Go to [Step 4](#step_4) to interactively run the script on the container.


Console in:

```bash
python launch_benchmark.py \
    --model-name ssd_vgg16 \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --model-source-dir ~/object_detection/ssd_vgg16/SSD.TensorFlow \
    --data-location ~/object_detection/ssd_vgg16/data_tfrecords \
    --in-graph ~/object_detection/ssd_vgg16/ssdvgg16_fp32_pretrained_model.pb \
    --batch-size 1 \
    --socket-id 0 \
    --num-inter-threads 11 \
    --num-intra-threads 21 \
    --data-num-inter-threads 21 \
    --data-num-intra-threads 28 \
    -- warmup-steps=100 steps=500

```

The logs are captured in a directory outside of the container. 

4. <a name="step_4"></a>If you want to run ```launch_benchmark.py``` interactively from within the docker container, add flag ```--debug```. This launches a docker container based on the ```--docker_image```,
performs necessary installs, runs the ```launch_benchmark.py``` script, and does not terminate the container process. 

Console in:		
```bash
python launch_benchmark.py \
    --model-name ssd_vgg16 \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --model-source-dir ~/object_detection/ssd_vgg16/SSD.TensorFlow \
    --data-location ~/object_detection/ssd_vgg16/data_tfrecords \
    --in-graph ~/object_detection/ssd_vgg16/ssdvgg16_fp32_pretrained_model.pb \
    --batch-size 1 \
    --socket-id 0 \
    --num-inter-threads 11 \
    --num-intra-threads 21 \
    --data-num-inter-threads 21 \
    --data-num-intra-threads 28 \
    --debug \
    -- warmup-steps=100 steps=500  
```
Console out:	
```bash
	lscpu_path_cmd = command -v lscpu
	lscpu located here: b'/usr/bin/lscpu'
	root@a78677f56d69:/workspace/benchmarks/common/tensorflow#
```
	
To rerun the benchmarking script, execute the ```start.sh``` bash script from your existing directory with the available flags, which in turn will run ```launch_benchmark.py```. For example, to rerun the script and compute accuracy, run with `ACCURACY_ONLY` flag set to True. To skip the run from reinstalling packages pass ```True``` to ```NOINSTALL```. 

```bash	
	chmod +x ./start.sh
```
```bash
	NOINSTALL=True ACCURACY_ONLY=True ./start.sh
```

All other flags will default to values passed in the first ```launch_benchmark.py``` that starts the container. [See here](/docs/general/tensorflow/LaunchBenchmark.md) to get the full list of flags. 
	
	
