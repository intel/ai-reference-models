## Run the model

Download the pretrained model from huggingface and set the `PRETRAINED_MODEL` environment
variable to point to the downloaded file.
```
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin -O pytorch_model.bin
export PRETRAINED_MODEL=$(pwd)/pytorch_model.bin
```

Once you have the pretarined model and have [built the container](#build-the-container),
use the `run.sh` script from the container package to run <model name> <mode> in docker.
Set environment variables to specify the precision to run, and an output directory.
By default, the `run.sh` script will run the `inference_realtime.sh` quickstart script.
To run a different script, specify the name of the script using the `SCRIPT` environment
variable.
```
# Navigate to the container package directory
cd <package dir>

# Set the required environment vars
export PRETRAINED_MODEL=<path to the downloaded model>
export PRECISION=<specify the precision to run>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with inference_realtime.sh quickstart script
./run.sh

# Use the SCRIPT env var to run a different quickstart script
SCRIPT=accuracy.sh ./run.sh
```
