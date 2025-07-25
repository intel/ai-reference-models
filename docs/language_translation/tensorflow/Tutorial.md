# Language Translation with Transformer-LT


## Goal
This tutorial will introduce CPU performance considerations of the deep learning Transformer-LT model for language translation and how to use Intel® Optimizations for TensorFlow to improve inference time on CPUs.
This tutorial will also provide code examples to use Intel Model Zoo's pretrained English to German model that can be copy/pasted for quick off-the-ground implementation on real data.

## Background
Language Translation with deep learning is a computationally expensive endeavor. This tutorial will show you how to reduce the inference runtime of your Transformer-LT network, a popular topology solution to translation.
It is based on an encoder-decoder architecture with an added attention mechanism. The encoder is used to encode the original sentence to a meaningful fixed-length vector, and the decoder is responsible for extracting the context data from the vector.
The encoder and decoder process the inputs and outputs, which are in the form of a time sequence.

In a traditional encoder/decoder model, each element in the context vector is treated equally. This is typically not the ideal solution.
For instance, when you translate the phrase “I travel by train” from English into Chinese, the word “I” has a greater influence than other words when producing its counterpart in Chinese.
Thus, the attention mechanism was introduced to differentiate contributions of each element in the source sequence to their counterpart in the destination sequence, through the use of a hidden matrix.
This matrix contains weights of each element in the source sequence when producing elements in the destination sequence.


##  Recommended Settings
In addition to TensorFlow optimizations that use the [Intel® oneAPI Deep Neural Network Library (Intel® oneDNN)](https://github.com/oneapi-src/oneDNN) to utilize instruction sets appropriately, the runtime settings also significantly contribute to improved performance.
Tuning these options to optimize CPU workloads is vital to optimize performance of TensorFlow on Intel® processors.
Below are the set of run-time options tested empirically on Transformer-LT and recommended by Intel:


| Run-time options  | Recommendations |
| ------------- | ------------- |
| Batch Size | 64. Regardless of the hardware  |
| Hyperthreading  | Enabled. Turn on in BIOS. Requires a restart. |
|intra_op_parallelism_threads |# physical cores |
|inter_op_parallelism_threads | 1 |
|NUMA Controls| --cpunodebind=0 --membind=0 |
|KMP_AFFINITY| KMP_AFFINITY=granularity=fine,verbose,compact,1,0|
|KMP_BLOCKTIME| 1 |
|OMP_NUM_THREADS |physical cores|

Note 1: [Refer](https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html) to learn more about the run time options.

Note 2: You can remove `verbose` from `KMP_AFFINITY` setting to avoid verbose output at runtime.

Run the following commands to get your processor information:

a. #physical cores per socket: `lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`

b. #all physical cores: `lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`

Below is a code snippet you can incorporate into your existing TensorFlow application to set the best settings.
You can either set them in the CLI or in the Python script. Note that inter and intra_op_parallelism_threads settings can only be set
in the Python script.

```bash
export OMP_NUM_THREADS=physical cores
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
os.environ["OMP_NUM_THREADS"]= <# physical cores>
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(<# physical cores>)
```

## Hands-on Tutorial
This section shows how to measure inference performance on Intel's Model Zoo pretrained model (or your pretrained model) by setting the above-discussed run time flags.
### FP32 inference

### Initial Setup

1. Clone IntelAI models and download into your home directory, skip this step if you already have Intel AI models installed.

```bash
cd ~
git clone https://github.com/IntelAI/models.git
```

2. Skip to step 3 if you already have a pretrained model or download the file `transformer_lt_official_fp32_pretrained_model.tar.gz` into your ~/transformer_LT_german location.
```
mkdir ~/transformer_LT_german
cd ~/transformer_LT_german
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/transformer_lt_official_fp32_pretrained_model.tar.gz
tar -xzvf transformer_lt_official_fp32_pretrained_model.tar.gz
```
Refer to the Transformer LT Official [README](/benchmarks/language_translation/tensorflow/transformer_lt_official) to get the latest location of the pretrained model.

3. After extraction, you should see the following folders and files in the `transformer_lt_official_fp32_pretrained_model` directory:
```
$ ls -l transformer_lt_official_fp32_pretrained_model/*

transformer_lt_official_fp32_pretrained_model/data:
total 1064
-rw-r--r--. 1 <user> <group> 359898 Feb 20 16:05 newstest2014.en
-rw-r--r--. 1 <user> <group> 399406 Feb 20 16:05 newstest2014.de
-rw-r--r--. 1 <user> <group> 324025 Mar 15 17:31 vocab.txt

transformer_lt_official_fp32_pretrained_model/graph:
total 241540
-rwx------. 1 <user> <group> 247333269 Mar 15 17:29 fp32_graphdef.pb

```
`newstest2014.en`: Input file with English text<br>
`newstest2014.de`: German translation of the input file for measuring accuracy<br>
`vocab.txt`: A dictionary of vocabulary<br>
`fp32_graphdef.pb`: Pretrained model

Or, if you have your own model/data, ensure the folder structure following the structure depicted below to run the pretrained model in Intel Model Zoo.

```
├─ transformer_LT_german
│	    ├── transformer_pretrained_model
│	    	 ├── data
│	         │   ├── newstest2014.en (Input file)
│	   	 │   ├── newstest2014.de (Reference file, this is optional)
│	         │   └── vocab.txt
│	         └── graph
│	    	     └── pretrained_model.pb
```
4. Install [Docker](https://docs.docker.com/install/) since the tutorial runs in a Docker container.

### Run inference

1. Pull the relevant Intel-optimized TensorFlow Docker image.
   Click here to find  all the [available](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html) docker images.
```bash
docker pull docker.io/intel/intel-optimized-tensorflow:latest
```
2. cd to the inference script directory in local IntelAI repo
```bash
cd ~/models/benchmarks
```
3. Run the Python script ``` launch_benchmark.py``` with the pretrained model.
```launch_benchmark.py``` script can be treated as an entry point to conveniently perform out-of-box high performance
inference on pretrained models trained of popular topologies.
The script will automatically set the recommended run-time options for supported topologies,
but if you choose to set your own options, refer to full of available flags and a detailed
explanation on ```launch_benchmarking.py``` [script](/docs/general/tensorflow/LaunchBenchmark.md).
 This step will automatically launch a new container on every run and terminate. Go to [Step 4](#step_4) to interactively run the script on the container.

3.1. <b> *Online inference*</b> (using `--socket-id 0` and `--batch-size 1`)

If you wish to calculate the [BLEU](https://en.wikipedia.org/wiki/BLEU) metric to find out the machine-translation quality, pass the file as `reference` flag.
`newstest2014.en` file must have only one sentence per line


console in:
```bash
python launch_benchmark.py \
     --model-name transformer_lt_official \
     --precision fp32 \
     --mode inference \
     --framework tensorflow \
     --batch-size 1 \
     --socket-id 0 \
     --docker-image intel/intel-optimized-tensorflow:latest \
     --in-graph ~/transformer_LT_german/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb \
     --data-location ~/transformer_LT_german/transformer_lt_official_fp32_pretrained_model/data \
     -- file=newstest2014.en \
     vocab_file=vocab.txt \
     file_out=translate.txt \
     reference=newstest2014.de
```

The translated German text will be in the file `translation.txt` located at `~/models/benchmarks/common/tensorflow/logs`

3.2. <b>*Batch inference*</b> (using `--socket-id 0` and `--batch-size 64`)

```bash
python launch_benchmark.py \
	--model-name transformer_lt_official \
	--precision fp32 \
	--mode inference \
	--framework tensorflow \
	--batch-size 64 \
	--socket-id 0 \
	--docker-image intel/intel-optimized-tensorflow:latest \
	--in-graph ~/transformer_LT_german/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb \
	--data-location ~/transformer_LT_german/transformer_lt_official_fp32_pretrained_model/data \
	-- file=newstest2014.en \
	vocab_file=vocab.txt \
	file_out=translate.txt \
	reference=newstest2014.de
```
console out:
```
Graph parsed in ..... s
import_graph_def took .....s
tokenizer took ..... s
Translating 3003 sentences from English to German.
Total inferencing time:....
Throughput:.... sentences/second
Total number of sentences translated:3003
I0419 22:50:49.856748 140013257643776 compute_bleu.py:106] Case-insensitive results: 27.510020
I0419 22:50:51.203501 140013257643776 compute_bleu.py:110] Case-sensitive results: 26.964748
Ran inference with batch size 64
Log location outside container: /~/models/benchmarks/common/tensorflow/logs/benchmark_transformer_lt_official_inference_fp32_20190419_224047.log
```

The logs are captured in a directory outside of the container.<br>

4. <a name="step_4"></a>If you want to run the ```launch_benchmark.py``` interactively from within the docker container, add flag ```--debug```. This will launch a docker container based on the ```--docker_image```,
performs necessary installs, runs the ```launch_benchmark.py``` script and does not terminate the container process. As an example, this step will demonstrate online inference (--batch-size 1), but you can implement the same strategy for batch inference (--batch-size 64)."

console in:
```bash
python launch_benchmark.py \
	--model-name transformer_lt_official \
	--precision fp32 \
	--mode inference \
	--framework tensorflow \
	--batch-size 64 \
	--socket-id 0 \
	--docker-image intel/intel-optimized-tensorflow:latest \
	--in-graph ~/transformer_LT_german/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb \
	--data-location ~/transformer_LT_german/transformer_lt_official_fp32_pretrained_model/data \
	--debug \
	-- file=newstest2014.en \
	vocab_file=vocab.txt \
	file_out=translate.txt \
	reference=newstest2014.de

```
console out:
```bash
	lscpu_path_cmd = command -v lscpu
	lscpu located here: b'/usr/bin/lscpu'
	root@a78677f56d69:/workspace/benchmarks/common/tensorflow#
```

To rerun the benchmarking script, execute the ```start.sh``` bash script from your existing directory with the available flags, which in turn will run ```launch_benchmark.py```. For e.g  to rerun with the different batch size (batch size=64) settings run with ```BATCH_SIZE```
and to skip the run from reinstalling packages pass ```True``` to ```NOINSTALL```.

```bash
	chmod +x ./start.sh
```
```bash
	NOINSTALL=True BATCH_SIZE=64 ./start.sh
```

All other flags will be defaulted to values passed in the first ```launch_benchmark.py``` that starts the container. [See here](/docs/general/tensorflow/LaunchBenchmark.md) to get the full list of flags.
