<!--- 30. Datasets -->
## Datasets

Use [prepare_data.sh](https://github.com/alibaba/ai-matrix/blob/master/macro_benchmark/DIEN_TF2/prepare_data.sh) to get [a subset of the Amazon book reviews data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/) and process it.
Or download and extract the preprocessed data files directly:
```
wget https://zenodo.org/record/3463683/files/data.tar.gz
wget https://zenodo.org/record/3463683/files/data1.tar.gz
wget https://zenodo.org/record/3463683/files/data2.tar.gz

tar -jxvf data.tar.gz
mv data/* .
tar -jxvf data1.tar.gz
mv data1/* .
tar -jxvf data2.tar.gz
mv data2/* .
```
Set the `DATASET_DIR` to point to the directory with the dataset files when running <model name>.
