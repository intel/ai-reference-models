## Datasets

The LibriSpeech dataset is used by RNN-T. Use the <model name> <mode> container
to download and prepare the training dataset. Specify a directory the dataset will be
downloaded to when running the container:

```
export DATASET_DIR=<folder where the training dataset will be downloaded>
mkdir -p $DATASET_DIR

docker run --rm \
  --env DATASET_DIR=${DATASET_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  -w /workspace/<package dir> \
  -it \
  <docker image> \
  /bin/bash quickstart/download_dataset.sh
```

This `DATASET_DIR` environment variable will be used again when
[running the model](#run-the-model).
