<!--- 30. Datasets -->
## Datasets and Pretrained Model

1. Download the [MS COCO 2014 dataset](http://cocodataset.org/#download). 

    Set the `DATASET_DIR` to point to this directory when running <model name>.

2. Clone the [Mask R-CNN model repository](https://github.com/matterport/Mask_RCNN).
It is used as external model directory for dependencies. Download pre-trained COCO weights `mask_rcnn_coco.h5)` from the
[Mask R-CNN repository release page](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5),
and place it in the `MaskRCNN` directory.
```
$ https://github.com/matterport/Mask_RCNN.git
$ cd Mask_RCNN
$ wget -q https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
```

    Set `MODEL_SRC_DIR` to path to `MaskRCNN` directory

