<!--- 30. Datasets -->
## Datasets

Download and extract the ImageNet2012 training and validation dataset from
[http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script and extracting the images, your folder structure
should look something like this:
```
imagenet
├── train
│   ├── n02085620
│   │   ├── n02085620_10074.JPEG
│   │   ├── n02085620_10131.JPEG
│   │   ├── n02085620_10621.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` and `train` directories should be set as the
`DATASET_DIR` environment variable before running the quickstart scripts.
