# Stable Diffusion Inference

Stable Diffusion Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    Pytorch    |       -        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU Max or Flex or Arc
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Prepare Dataset
the scripts will download the dataset automatically. it using nateraw/parti-prompts (https://huggingface.co/datasets/nateraw/parti-prompts) as the dataset. 

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/stable_diffusion/inference/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=PVC` (PVC or ATS-M or ARC)                                                 |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=1`                                |
| **PRECISION** (optional)     |           `export PRECISION=fp16` (fp16 and fp32 are supported for all platform)|
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
6. Run `run_model.sh`

## Output

Single-tile output will typicall looks like:

```
No policy available for current head_size 512
inference Latency: 3671.8995094299316 ms
inference Throughput: 0.2723386076966065 samples/s
CLIP score: 33.59451
```

Multi-tile output will typicall looks like:
```
26%|██▌       | 13/50 [00:00<00:01, 20.13it/s]inference Latency: 3714.4706646601358 ms
inference Throughput: 0.2692173637320938 samples/s
CLIP score: 33.58945666666667
100%|██████████| 50/50 [00:02<00:00, 19.64it/s]
No policy available for current head_size 512
inference Latency: 3794.5287148157754 ms
inference Throughput: 0.26353733893104825 samples/s
CLIP score: 33.58307666666666
```
please noted that we have using it/s as the throughput. you can find in the `results.yaml`.

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 41.4400
   unit: it/s
it/s
 - key: latency
   value: 0.0482633
   unit: s
 - key: accuracy
   value: 33.5335
   unit: accuracy
```
