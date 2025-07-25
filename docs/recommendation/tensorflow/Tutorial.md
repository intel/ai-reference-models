# Recommendation System with Wide and Deep Model


## Goal
This tutorial will introduce CPU performance considerations for the popular [Wide and Deep](https://arxiv.org/abs/1606.07792) model to solve recommendation system problems
and how to tune run-time parameters to maximize performance using Intel® Optimizations for TensorFlow.
This tutorial also includes a hands-on demo on Intel Model Zoo's Wide and Deep pretrained model built using a dataset from [Kaggle's Display Advertising Challenge](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
to run online (real-time) and batch inference.

## Background
Google's latest innovation  to solve some of the shortcomings in traditional recommendation systems is the
Wide and Deep model, which combines the best aspects of linear modeling and deep neural networks to
outperform either approach individually by a significant percentage. In practice, linear models follow the simple mechanism of capturing feature relationships, resulting in
a lot of derived features. This piece of the topology is called “wide” learning, while the complexities
to generalize these relationships are solved by the "deep" piece of this topology. This wide and deep combination has
proven to be a robust approach in handling the underfitting and overfitting problems caused by unique feature combinations, however
at the cost of significant compute power.
Google has published a blog on [Wide and Deep learning with TensorFlow](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html),
and Datatonic has published [performance gains of CPU over GPU](https://datatonic.com/insights/accelerate-machine-learning-on-google-cloud-with-intel-xeon-processors/) for these types of models.

## Recommended Settings
Although there is no one-size-fits-all solution to maximize Wide and Deep model performance on CPUs, understanding the bottlenecks and tuning the run-time
parameters based on your dataset and TensorFlow graph can be extremely beneficial.

A recommendation system with Wide and Deep model topology comes with two main caveats:
Unlike image recognition models such as ResNet50 or ResNet101, the "wide" component of this model performs more “independent” operations and
does not provide opportunities to exploit parallelism within each node, while, the "deep" component of this topology demands more parallelism within each node.

The wide or linear component of this topology depends on the data features, i.e. on the dataset width.
The deep component depends on the number of hidden units in the graph where threading can be enabled within the operations,
hence exhibiting a direct relation to compute power. This setback can be eliminated by setting the right number of intra_op_threads and OMP_NUM_THREADS.
Note that while tuning these important run-time parameters, do not over/under use the threadpool.

| Run-time options  | Recommendations|
| ------------- | ------------- |
| Batch Size | 512 |
| Hyperthreading  | Enabled. Turn on in BIOS. Requires a restart. |
|intra_op_parallelism_threads| 1 to physical cores |
|inter_op_parallelism_threads | 1 |
|Data Layout| NC|
|Sockets | all |
|KMP_AFFINITY| granularity=fine,noverbose,compact,1,0|
|KMP_BLOCKTIME| 1 |
|OMP_NUM_THREADS | 1 to physical cores |

*Note: Refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html) to learn more about the run time options.*

Intel's data science team trained and published a Wide and Deep model on Kaggle's Display Advertising Challenge dataset, and has empirically tested and identified the best run-time settings
to run inference, which is illustrated below in the hands-on-tutorial section.

Run the following commands to get your processor information:

a. #physical cores per socket:
`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`

b. #all physical cores:
`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`

Below is a code snippet you can incorporate into your Wide and Deep TensorFlow application to start tuning towards the best settings.
You can either set them in the CLI or in the Python script. Note that inter and intra_op_parallelism_threads settings can only be set
in the Python script.

```bash
export OMP_NUM_THREADS=no of physical cores
export KMP_AFFINITY="granularity=fine,noverbose,compact,1,0"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
```
(or)
```
import os
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"]= no of physical cores
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(<# physical cores>)
```


To control the execution to one NUMA node or socket id, run the python script with the command:

```
numactl --cpunodebind=0  --membind=0 python <application script> <args>
```

## Hands-on Tutorial


This section shows how to measure inference performance on Intel's Model Zoo Wide and Deep pretrained model trained
on Kaggle's Display Advertising Challenge dataset by setting the above-discussed run time flags.

### FP32 inference

### Initial Setup
1. Clone IntelAI models and download into your home directory.

```bash
git clone https://github.com/IntelAI/models.git
```

2. Download the pretrained model ```wide_deep_fp32_pretrained_model.pb``` into your `~/wide_deep_files` location.

```
mkdir ~/wide_deep_files
cd ~/wide_deep_files
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.pb

```
Refer to the Wide and Deep [README](/benchmarks/recommendation/tensorflow/wide_deep_large_ds) to get the latest location of the pretrained model.

3. Install [Docker](https://docs.docker.com/install/) since the tutorial runs on a Docker container.


4. Data Preparation: You will need approximately 20GB of available disk space to complete this step.
Follow the instructions below to download and prepare the dataset.
  - Prepare the data directory:
    ```
    mkdir ~/wide_deep_files/real_dataset
    cd ~/wide_deep_files/real_dataset
    ```

  - Download the eval set:

    ```wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv```

  - Move the downloaded dataset to `~/models/models` and start a Docker container for preprocessing:
    ```
    mv eval.csv ~/models/models
    cd ~/models/models
    docker run -it --privileged -u root:root \
            -w /models \
            --volume $PWD:/models \
            intel/intel-optimized-tensorflow:1.15.2 \
            /bin/bash
    ```
  - Preprocess and convert eval dataset to TFRecord format. We will use a script in the Intel Model Zoo repository.
    This step may take a while to complete.
    ```
    python recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
        --inputcsv-datafile eval.csv \
        --calibrationcsv-datafile train.csv \
        --outputfile-name preprocessed
    ```
  - Exit the docker container and find the processed dataset `eval_preprocessed.tfrecords` in the location `~/models/models`.

### Run inference

1. Pull the relevant Intel Optimizations for TensorFlow Docker image. We'll be running the pretrained model to infer in a Docker container.
   Click here to find  all the [available](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html) docker images.
```bash
docker pull intel/intel-optimized-tensorflow:latest
```
2. cd to the inference script directory:
```bash
cd ~/models/benchmarks
```
3. Run the Python script ``` launch_benchmark.py``` with the pretrained model.
The ```launch_benchmark.py``` script can be treated as an entry point to conveniently perform out-of-box high performance
inference on pretrained models of popular topologies.
The script will automatically set the recommended run-time options for supported topologies,
but if you choose to set your own options, refer to the full list of available flags and a detailed
explanation of the ```launch_benchmark.py``` [script](/docs/general/tensorflow/LaunchBenchmark.md).
This step will automatically launch a new container on every run and terminate. Go to [Step 4](#step_4) to interactively run the script in the container.

&nbsp;&nbsp;&nbsp;&nbsp;3.1. <b> *Online Inference*</b> (also called real-time inference, batch_size=1)

Note: As per the recommended settings `socket-id` is set to -1 to run on all sockets.
Set this parameter to a socket id to run the workload on a single socket.


	python launch_benchmark.py \
        --batch-size 1 \
        --model-name wide_deep_large_ds \
        --precision fp32 \
        --mode inference \
        --framework tensorflow \
        --benchmark-only \
        --docker-image intel/intel-optimized-tensorflow:latest \
        --in-graph ~/wide_deep_files/wide_deep_fp32_pretrained_model.pb \
        --data-location ~/models/models/eval_preprocessed.tfrecords \
        --verbose

&nbsp;&nbsp;&nbsp;&nbsp;3.2. <b>*Batch Inference*</b> (batch_size=512)

Note: As per the recommended settings `socket-id` is set to -1 to run on all sockets.
Set this parameter to a socket id to run the workload on a single socket.

	python launch_benchmark.py \
        --batch-size 512 \
        --model-name wide_deep_large_ds \
        --precision fp32 \
        --mode inference \
        --framework tensorflow \
        --benchmark-only \
        --docker-image intel/intel-optimized-tensorflow:latest \
        --in-graph ~/wide_deep_files/wide_deep_fp32_pretrained_model.pb \
        --data-location ~/models/models/eval_preprocessed.tfrecords \
        --verbose

&nbsp;&nbsp;&nbsp;&nbsp;<u>Example Output:</u>

	--------------------------------------------------
	Total test records           :  2000000
	Batch size is                :  512
	Number of batches            :  3907
	Inference duration (seconds) :  ...
	Average Latency (ms/batch)   :  ...
	Throughput is (records/sec)  :  ...
	--------------------------------------------------
	num_inter_threads: 28
	num_intra_threads: 1
	Received these standard args: Namespace(accuracy_only=False, batch_size=512, benchmark_dir='/workspace/benchmarks', benchmark_only=True, checkpoint=None, data_location='/dataset', data_num_inter_threads=None, data_num_intra_threads=None, framework='tensorflow', input_graph='/in_graph/wide_deep_fp32_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='wide_deep_large_ds', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=28, num_intra_threads=1, output_dir='/workspace/benchmarks/common/tensorflow/logs', output_results=False, precision='fp32', socket_id=-1, use_case='recommendation', verbose=True)
	Received these custom args: []
	Current directory: /workspace/benchmarks
	Running: python /workspace/intelai_models/inference/inference.py --num_intra_threads=1 --num_inter_threads=28 --batch_size=512 --input_graph=/in_graph/wide_deep_fp32_pretrained_model.pb --data_location=/dataset
	PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
	RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=recommendation --model-name=wide_deep_large_ds --precision=fp32 --mode=inference --model-source-dir=/workspace/models --benchmark-dir=/workspace/benchmarks --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=512 --socket-id=-1 --output-dir=/workspace/benchmarks/common/tensorflow/logs  --benchmark-only  --verbose --in-graph=/in_graph/wide_deep_fp32_pretrained_model.pb --data-location=/dataset
	Batch Size: 512
	Ran inference with batch size 512
	Log location outside container: /home/<user>/models/benchmarks/common/tensorflow/logs/benchmark_wide_deep_large_ds_inference_fp32_20190316_164924.log

The logs are captured in a directory outside of the container.<br>

&nbsp;&nbsp;&nbsp;&nbsp;3.3. <b> *Compute accuracy on eval dataset*</b>

	python launch_benchmark.py \
        --batch-size 512 \
        --model-name wide_deep_large_ds \
        --precision fp32 \
        --mode inference \
        --framework tensorflow \
        --accuracy-only \
        --docker-image intel/intel-optimized-tensorflow:latest \
        --in-graph ~/wide_deep_files/wide_deep_fp32_pretrained_model.pb \
        --data-location ~/models/models/eval_preprocessed.tfrecords \
        --verbose

   &nbsp;&nbsp;&nbsp;&nbsp;<u>Example Output:</u>

With `accuracy-only` flag, you can find an additional metric on accuracy as shown in the output below

	--------------------------------------------------
	Total test records           :  2000000
	Batch size is                :  512
	Number of batches            :  3907
	Classification accuracy (%)  :  77.6693
	No of correct predictions    :  ...
	Inference duration (seconds) :  ...
	Average Latency (ms/batch)   :  ...
	Throughput is (records/sec)  :  ...
	--------------------------------------------------

4. <a name="step_4"></a>If you want to run the benchmarking script interactively within the docker container, run ```launch_benchmark.py``` with ```--debug``` flag. This will launch a docker container based on the ```--docker_image```,
perform necessary installs, run the ```launch_benchmark.py``` script, and does not terminate the container process.

		python launch_benchmark.py \
			--batch-size 1 \
			--model-name wide_deep_large_ds \
			--precision fp32 \
			--mode inference \
			--framework tensorflow \
			--benchmark-only \
			--docker-image intel/intel-optimized-tensorflow:latest \
			--in-graph ~/wide_deep_files/wide_deep_fp32_pretrained_model.pb \
			--data-location ~/models/models/eval_preprocessed.tfrecords \
			--debug

&nbsp;&nbsp;<u>Example Output:</u>

	root@a78677f56d69:/workspace/benchmarks/common/tensorflow#

To rerun the model script, execute the ```start.sh``` bash script from your existing directory with additional or modified flags. For example, to rerun with the best batch inference (batch size=512) settings, run with ```BATCH_SIZE```
and to skip the run from reinstalling packages pass ```True``` to ```NOINSTALL```.

	chmod +x ./start.sh

	NOINSTALL=True BATCH_SIZE=512 SOCKET_ID=0 VERBOSE=True ./start.sh

All other flags will be defaulted to values passed in the first ```launch_benchmark.py``` that starts the container. [See here](/docs/general/tensorflow/LaunchBenchmark.md) to get the full list of flags.

5. <b> Inference on a large dataset (optional) </b>

To run inference on a large dataset, download the test dataset in `~/wide_deep_files/real_dataset`. Note that this dataset supports only `benchmark-only` flag.

```
cd ~/wide_deep_files/real_dataset
```

- Go to this [page](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) on the Criteo website.
Agree to the terms of use, enter your name, and submit the form. Then copy the download link for the 4.3GB tar file called `dac.tar.gz` and use it in the `wget` command in the code block below.
Untar the file to create three files:
    1. readme.txt
    2. train.txt (11GB) - you will not be using this, so delete it to save space
    3. test.txt (1.4GB) - transform this into .csv

    ```
    wget <download_link> # replace <download_link> with link you got from Criteo website
    tar -xvf dac.tar.gz
    rm train.txt
    tr '\t' ',' < test.txt > test.csv
    ```

- Move the downloaded dataset to `~/models/models` and start a Docker container for preprocessing. This step is similar to `eval` dataset preprocessing:

    ```
    mv test.csv ~/models/models
    cd ~/models/models
    docker run -it --privileged -u root:root \
            -w /models \
            --volume $PWD:/models \
            intel/intel-optimized-tensorflow:latest \
            /bin/bash
    ```

- Preprocess and convert test dataset to TFRecord format. We will use a script in the Intel Model Zoo repository.
    This step may take a while to complete

	```
    python recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
        --inputcsv-datafile test.csv \
        --outputfile-name preprocessed
    ```

- Exit the docker container and find the processed dataset `test_preprocessed.tfrecords` in the location `~/models/models`.

&nbsp;&nbsp;&nbsp;&nbsp;5.1. <b>*Batch or Online Inference*</b>

	cd ~/models/benchmarks

	python launch_benchmark.py \
			--batch-size 512 \
			--model-name wide_deep_large_ds \
			--precision fp32 \
			--mode inference \
			--framework tensorflow \
			--benchmark-only \
			--docker-image intel/intel-optimized-tensorflow:latest \
			--in-graph ~/wide_deep_files/wide_deep_fp32_pretrained_model.pb \
			--data-location ~/models/models/test_preprocessed.tfrecords \
			--verbose

Set batch_size to 1 to run for online (real-time) inference
