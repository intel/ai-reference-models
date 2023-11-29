# Stable Diffusion Inference

Stable Diffusion Inference BKC.

## Model Information

| **Use Case** | **Framework** | **Model Repo** |          **Branch/Commit/Tag**           |  **Optional Patch** |
|:---:| :---: | :---: |:----------------------------------------:| :---: |
|  Inference   |  Tensorflow   | [keras-cv](https://github.com/keras-team/keras-cv.git) | 66fa74b6a2a0bb1e563ae8bce66496b118b95200 |  [patch](#patch) |

# Pre-Requisite
* Host has Intel® Data Center GPU FLEX
* Host has installed latest Intel® Data Center GPU Flex Series Driver https://dgpu-docs.intel.com/driver/installation.html


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models/diffusion/tensorflow/stable_diffusion/inference/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

    |   **Parameter**    |                   **export command**                    |
    |:-------------------------------------------------------:| :---: |
    |   **PRECISION**   |     `export PRECISION=fp16` (fp32 or fp16)      |
6. Run `run_model.sh`

## Output

Output will typicall looks like:
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
