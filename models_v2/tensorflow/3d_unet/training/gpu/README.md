# Unet-3D Model Training for Intel® Extention for TensorFlow
Best known method of Unet-3D training for Intel® Extention for TensorFlow.

## Model Information
| **Use Case** |**Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch**
| :---: | :---: | :---: | :---: | :---: |
| Training | TensorFlow | [DeepLearningExamples/UNet_3D](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical) | master | 3dunet_itex.patch <br /> 3dunet_itex_with_horovod.patch |

# Pre-Requisite
* Host has Intel® Data Center GPU Max
* Host has installed latest Intel® Data Center GPU Max Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Dataset 
```
The 3D-UNet model was trained in the [Brain Tumor Segmentation 2019 dataset](https://www.med.upenn.edu/cbica/brats2019/data.html). Test images provided by the organization were used to produce the resulting masks for submission. Upon registration, the challenge's data is made available through the https//ipp.cbica.upenn.edu service.
```

## Run Model
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/3d_unet/training/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

    |   **Parameter**    | **export command**                                    |
    | :---: | :--- |
    |  **DATASET_DIR**   | `export DATASET_DIR=/the/path/to/dataset`             |
    |   **OUTPUT_DIR**   | `export OUTPUT_DIR=/the/path/to/output_dir`           |
    |   **MULTI_TILE**   | `export MULTI_TILE=False (False or True)`           |
    |   **BATCH_SIZE** (optional)   | `export BATCH_SIZE=1`           |
    |   **PRECISION**   | `export PRECISION=bfloat16` (bfloat16 or fp32)           |
6. Run `run_model.sh`

## Output

Output will typicall looks like:
```
Current step: 997, time in ms: 73.95
Current step: 998, time in ms: 74.81
Current step: 999, time in ms: 74.31
Current step: 1000, time in ms: 74.44
self._step: 1000
Total time spent (after warmup): 79390.14 ms
Time spent per iteration (after warmup): 79.55 ms
Latency is 79.549 ms
Throughput is xxx samples/sec
```

Final results of the training run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: xxx
   unit: images/sec
```
