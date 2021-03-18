<!--- 20. Datasets -->
## Datasets

The [ImageNet](http://www.image-net.org/) validation dataset is used when
testing accuracy. The inference scripts use synthetic data, so no dataset
is needed.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/,
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

The accuracy script looks for a folder named `val`, so after running the
data prep script, your folder structure should look something like this:

```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the
`DATASET_DIR` when running accuracy
(for example: `export DATASET_DIR=/home/<user>/imagenet`).
