## Run the model

After you've followed the instructions to [build the container](#build-the-container),
download the [dataset and config file](#datasets), use the `run.sh` script from the
container package to run BERT Large training in docker. Set environment variables to
specify the precision to run, dataset directory, config file directory, and an
output directory. Use an empty `OUTPUT_DIR` to start with to prevent any previously
generated checkpoints from getting picked up. By default, the `run.sh` script will
run the `run_bert_pretrain_phase1.sh` quickstart script.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars to run phase 1
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log and model files will be written for phase 1>
export DATASET_DIR=<path to the preprocessed dataset>
export CONFIG_FILE=<path to the bert_config.json>

# Run the container with the default run_bert_pretrain_phase1.sh quickstart script
./run.sh
```
To run phase 2, to the directory where checkpoints were generated during phase 1
pretraining. This `CHECKPOINT_DIR` should also have the `bert_config.json` file.
```
# To run phase 2, set the CHECKPOINT_DIR to the folder with checkpoints generated during phase 1
export CHECKPOINT_DIR=<directory with checkpoints and the bert_config.json file>

# Define a new directory for phase 2 output, and set the SCRIPT var to run phase 2
export OUTPUT_DIR=<directory where log and model files will be written for phase 2>
SCRIPT=run_bert_pretrain_phase2.sh ./run.sh
```
