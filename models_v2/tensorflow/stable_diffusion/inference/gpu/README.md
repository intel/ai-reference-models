# Stable Diffusion Inference

Stable Diffusion Inference using Intel® Extension for TensorFlow.

## Model Information

| **Use Case** | **Framework** | **Model Repo** |          **Branch/Commit/Tag**           |  **Optional Patch** |
|:---:| :---: | :---: |:----------------------------------------:| :---: |
|  Inference   |  Tensorflow   | [keras-cv](https://github.com/keras-team/keras-cv.git) | 66fa74b6a2a0bb1e563ae8bce66496b118b95200 |  [patch](#patch) |

**Note**: Refer to [CONTAINER.md](CONTAINER.md) for Stable Diffusion Inference instructions using docker containers.

# Pre-Requisite
* Host has Intel® Data Center GPU Flex Series
* Host has installed latest Intel® Data Center GPU Flex Series Driver https://dgpu-docs.intel.com/driver/installation.html
* Install [Intel® Extension for TensorFlow](https://pypi.org/project/intel-extension-for-tensorflow/)


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/stable_diffusion/inference/gpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install [tensorflow and ITEX](https://pypi.org/project/intel-extension-for-tensorflow/)
6. Setup required environment paramaters

    |   **Parameter**    |                   **export command**                    |
    |:-------------------------------------------------------:| :---: |
    |   **PRECISION**   |     `export PRECISION=fp16` (fp32 or fp16)      |
7. Run `run_model.sh`

## Output

Output typically looks like:
```
50/50 [==============================] - 8s 150ms/step
latency 153.37058544158936 ms, throughput 6.520155068331838 it/s
Start plotting the generated images to ./images/fp16_imgs_50steps.png
```

Final results of the training run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 6.520155068331838
   unit: it/s
```
