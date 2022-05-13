## Datasets

### Input files

Download the following files from the [MLCommons members Google Drive](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT):
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

### Download and extract the dataset

From the [MLCommons BERT Processed dataset
directory](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v?usp=sharing)
download the `results_text.tar.gz` and `bert_reference_results_text_md5.txt` files. Perform the
following steps to extract the tar file and check the MD5 checksums.
```shell
tar xf results_text.tar.gz
cd results4
md5sum --check ../bert_reference_results_text_md5.txt
cd ..
```
After completing this step you should have a directory called `results4/` that
contains 502 files for a total of about 13Gbytes.

### Generate the BERT input dataset

There are input preprocessing scripts in the `<package dir>/input_preprocessing` folder.
Note that the pretraining dataset is about `539GB`, so ensure that you have enough disk
space. First, navigate to the folder with the preprocessing scripts.
```
cd <package dir>/input_preprocessing
```
> Note that the `results4` folder and the `vocab.txt` file will need to be in the
> `input_preprocessing` before running the preprocessing scripts.

The `create_pretraining_data.py` script duplicates the input plain text, replaces
different sets of words with masks for each duplication, and serializes the
output into the HDF5 file format.

The following snippet shows how `create_pretraining_data.py` is called by a parallelized
script that can be called as shown below.  The script reads the text data from
the `results4/` subdirectory and outputs the resulting 500 hdf5 files to a
subdirectory named `hdf5/`.

```shell
pip install tensorflow-cpu
```

For phase1 the seq_len=128:
```shell
export SEQ_LEN=128
./parallel_create_hdf5.sh
```
For phase2 the seq_len=512:
```shell
export SEQ_LEN=512
./parallel_create_hdf5.sh
```

The resulting `hdf5` subdirs will have 500 files named
`part-00???-of-0500.hdf5` and have a size of about 539 Gigabytes.

Next we need to shard the data into 2048 chunks.  This is done by calling the
chop_hdf5_files.py script.  This script reads the 500 hdf5 files from
subdirectory `hdf5` and creates 2048 hdf5 files in subdirectory
`2048_shards_uncompressed`.

For phase1:

```shell
export SEQ_LEN=128
python3 ./chop_hdf5_files.py
```

For phase2:

```shell
export SEQ_LEN=512
python3 ./chop_hdf5_files.py
```

The above will produce a subdirectory named `2048_shards_uncompressed/`
containing 2048 files named `part_*_of_2048.hdf5` and have a size of about 539 Gigabytes.
you can use "SHARD_NUM" to control the shard files number. the default "SHARD_NUM" if 2048.

```
<DATASET_DIR>
├── 2048_shards_uncompressed_512
│   └── part-00000-of-00xxx
└── 2048_shards_uncompressed_128
    └── part-00000-of-00xxx
```
