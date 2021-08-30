## Run the model

After you've followed the instructions to [build the container](#build-the-container),
download the [dataset and config file](#datasets), use the `run.sh` script from the
container package to run <model name> <mode> in docker. Set environment variables to
specify the precision to run, dataset directory, config file directory, and an
output directory. By default, the `run.sh` script will run the `pretrain_phase1.sh`
quickstart script.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars to run phase 1
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log and model files will be written for phase 1>
export DATASET_DIR=<path to the preprocessed dataset>
export CONFIG_FILE=<path to the bert_config.json>

# Run the container with the default pretrain_phase1.sh quickstart script
./run.sh
```
To run phase 2, use the model_save from the phase 1 output as the `CHECKPOINT_DIR`.
Alternatively, checkpoints can be downloaded from online (for example, from the
[MLCommons repo](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert),
but these are TF checkpoints that would need to be converted to a PyTorch model file).
```
# To run phase 2, set the CHECKPOINT_DIR to the model_save directory from phase 1's output
export CHECKPOINT_DIR=${OUTPUT_DIR}/model_save

# Define a new directory for phase 2 output, and set the SCRIPT var to run phase 2
export OUTPUT_DIR=<directory where log and model files will be written for phase 2>
SCRIPT=pretrain_phase2.sh ./run.sh
```
