## Run the model

Download the `config.json` and fine tuned model from huggingface and set the `PRETRAINED_MODEL`
environment variable to point to the directory that has both files:
```
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
PRETRAINED_MODEL=$(pwd)/bert_squad_model
```

Once you have the pretarined model and have [built the container](#build-the-container),
use the `run.sh` script from the container package to run <model name> <mode> in docker.
Set environment variables to specify the precision to run, and an output directory.
By default, the `run.sh` script will run the `run_multi_instance_realtime.sh` quickstart script.
To run a different script, specify the name of the script using the `SCRIPT` environment
variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRETRAINED_MODEL=<path to the downloaded model files>
export EVAL_DATA_FILE=<path to the dev-v1.1.json file>
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Use the SCRIPT env var to run a different quickstart script
SCRIPT=run_accuracy.sh ./run.sh
```
