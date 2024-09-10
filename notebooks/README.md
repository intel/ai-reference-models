# Run Intel® AI Reference Models in a Jupyter Notebook

This Jupyter notebook helps you choose and run a comparison between two models from the [Intel® AI Reference Models repo](https://github.com/IntelAI/models) using Intel® Optimizations for TensorFlow*. When you run the notebook, it installs required package dependencies, displays information about your platform, lets you choose the two models to compare, runs those models, and finally displays a performance comparison chart.

## Supported Models

| Model | Framework | Mode |  Supported Precisions |
| ----- | --------- | ---- | ----------- | -------------------- |
| ResNet50v1.5 | TensorFlow | Inference | [FP32, BFloat32, FP16, INT8, BFloat16](/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/README.md) |
| BERT Large | TensorFlow | Inference | [FP32, BFLOAT16, FP16](/benchmarks/language_modeling/tensorflow/bert_large/inference/README.md)

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
