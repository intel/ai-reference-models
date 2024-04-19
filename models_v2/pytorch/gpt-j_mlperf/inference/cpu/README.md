# GPT-J MLPerf Inference

GPT-J MLPerf Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    Pytorch    |   https://github.com/mlcommons/inference/tree/master/language/gpt-j#download-gpt-j-model       |           -           |         -          |

## Pre-Requisite
  * **Intel 4th Generation Xeon Processor or later** - [4th Gen Intel® Xeon® Scalable Processors](https://ark.intel.com/content/www/us/en/ark/products/series/228622/4th-gen-intel-xeon-scalable-processors.html)
  * rclone installation is required. 

## Prepare Dataset
### Dataset:  cnn-dailymail 
cnn-dailymail dataset is used. setup.sh script will download dataset for users automatically. 

## (Optional) Pre-trained Model 

First, export the dataset folder path. 
ex: ~/Dataset (Users could use different dataset path) 
```bash
export DATA_DIR=~/Dataset
```  
Download the pre-trained model under the DATA_DIR folder to by pass quantization step.
```bash
mkdir -p ${DATA_DIR}/gpt-j/data/gpt-j-int4-model/
cd ${DATA_DIR}/gpt-j/data/gpt-j-int4-model/
wget https://storage.googleapis.com/intel-optimized-pytorch/models/mlperf/4.0/best_int4_model.pt
```
## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/gpt-j_mlperf/inference/cpu`
3. Create a new conda environment with the following command, replacing <your-env-name> with your preferred name for the environment:
    ```
    conda create -n <your-env-name>
    conda activate <your-env-name>
    ```
4. Install rclone for data downloading with below command:
    ```
    sudo -v ; curl https://rclone.org/install.sh | sudo bash  
    ```
5. Setup required environment paramaters

    | **Parameter**                |                                  **export command**                                  |
    |:---------------------------:|:------------------------------------------------------------------------------------:|
    | **DATA_DIR** (optional)    |    `export DATA_DIR=~/Dataset` (Users could use different dataset path)                                |
    | **OUTPUT_DIR** (optional)     |`export OUTPUT_DIR=~/Output` (Users could use different output path)  |

6. Run setup.sh  
    ```
    ./setup.sh
    ```  
7. Run
    ```
    ./run_model.sh
    ```  

## Output

Single-tile output will typically look like:

```
[2024-03-04 00:13:38,396][INFO] run.py:341  - ===== Performing gptj-99/pytorch-cpu/int4/Offline/performance =====


[2024-03-04 00:45:30,516][INFO] run.py:341  - ********************************************************************************

[2024-03-04 00:45:30,516][INFO] run.py:341  - gptj-99/pytorch-cpu/int4/Offline/performance:

[2024-03-04 00:45:30,516][INFO] run.py:341  -   Target QPS: 1.2

[2024-03-04 00:45:30,516][INFO] run.py:341  -   Perf QPS: 1.01455

[2024-03-04 00:45:30,516][INFO] run.py:341  -   99.00 percentile latency: 591394693466.0

[2024-03-04 00:45:30,516][INFO] run.py:341  -   Result dir: /output/closed/Intel/results/1-node-2S-EMR-PyTorch-INT4/gptj-99/Of                                                                                fline/performance/run_1

[2024-03-04 00:45:30,516][INFO] run.py:341  - ********************************************************************************


[2024-03-04 00:45:30,516][INFO] run.py:341  - ===== Performing gptj-99/pytorch-cpu/int4/Server/performance =====


[2024-03-04 08:06:27,283][INFO] run.py:341  - ********************************************************************************

[2024-03-04 08:06:27,283][INFO] run.py:341  - gptj-99/pytorch-cpu/int4/Server/performance:

[2024-03-04 08:06:27,283][INFO] run.py:341  -   Target QPS: 0.52

[2024-03-04 08:06:27,283][INFO] run.py:341  -   Perf QPS: 0.52

[2024-03-04 08:06:27,283][INFO] run.py:341  -   99.00 percentile latency: 394591832561.0

[2024-03-04 08:06:27,283][INFO] run.py:341  -   Result dir: /output/closed/Intel/results/1-node-2S-EMR-PyTorch-INT4/gptj-99/Se                                                                                rver/performance/run_1

[2024-03-04 08:06:27,283][INFO] run.py:341  - ********************************************************************************

```


Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 1.01455
   unit: it/s

```
