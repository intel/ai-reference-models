<!--- 30. Datasets -->
## Datasets and Pretrained Model

Download the [MS COCO 2014 dataset](http://cocodataset.org/#download).
Set the `DATASET_DIR` to point to this directory when running <model name>.
```
# Create a new directory, to be set as DATASET_DIR
mkdir $DATASET_DIR
cd $DATASET_DIR

# Download and extract MS COCO 2014 dataset
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
cp annotations/instances_val2014.json annotations/instances_minival2014.json

export DATASET_DIR=${DATASET_DIR}
```
