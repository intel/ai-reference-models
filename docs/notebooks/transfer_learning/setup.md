# Environment Setup and Running the Notebooks

Use the instructions below to install the dependencies required to run the notebooks.

## Set Up Notebook Environment

1. Get a clone of the Model Zoo repository from GitHub:
   ```
   git clone https://github.com/IntelAI/models.git intelai_models
   export MODEL_ZOO_DIR=$(pwd)/intelai_models
   ```
2. Create a Python3 virtual environment and install the required packages.
   
   You can use virtualenv:
   ```
   python3 -m venv intel-pyt-venv
   source intel-pyt-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n intel-pyt python=3.9
   conda activate intel-pyt
   ```
   Then, from inside the activated virtualenv or conda environment run these steps:
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   poetry install --directory ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning
   ```
3. Set environment variables for the path to the dataset folder and an output directory.
   The dataset and output directories can be empty. The notebook will download the
   dataset to the dataset directory, if it is empty. Subsequent runs will reuse the dataset.
   ```
   export DATASET_DIR=<directory to download the dataset>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server)

   ```
   cd ${MODEL_ZOO_DIR}/docs/notebooks/transfer_learning/
   jupyter notebook --NotebookApp.iopub_msg_rate_limit=10000.0 --NotebookApp.iopub_data_rate_limit=1e10 --port 8888
   
   ```

5. Copy and paste the URL from the terminal to your browser to view and run the notebooks.

Once you have the environment and dependencies set up, see the list of available
[notebooks](/docs/notebooks/transfer_learning/README.md).
