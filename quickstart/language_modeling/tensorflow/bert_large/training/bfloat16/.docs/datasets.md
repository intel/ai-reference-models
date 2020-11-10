<!--- 30. Datasets -->
## Datasets

### Pretrained models

Download and extract checkpoints the bert pretrained model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
The extracted directory should be set to the `CHECKPOINT_DIR` environment
variable when running the quickstart scripts.

For training from scratch, Wikipedia and BookCorpus need to be downloaded
and pre-processed.

### GLUE data

[GLUE data](https://gluebenchmark.com/tasks) is used when running BERT
classification training. Download and unpack the GLUE data by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e).

### SQuAD data

The Stanford Question Answering Dataset (SQuAD) dataset files can be downloaded
from the [Google bert repo](https://github.com/google-research/bert#squad-11).
The three files (`train-v1.1.json`, `dev-v1.1.json`, and `evaluate-v1.1.py`)
should be downloaded to the same directory. Set the `DATASET_DIR` to point to
that directory when running bert fine tuning using the SQuAD data.

