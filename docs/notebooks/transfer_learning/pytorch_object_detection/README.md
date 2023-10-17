# Transfer Learning for Object Detection using PyTorch

This notebook uses object detection models from torchvision that were originally trained
using COCO and does transfer learning with the [PennFudan dataset](https://www.cis.upenn.edu/~jshi/ped_html/),
available via public download, or the [Kitti dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d),
available through Torchvision datasets.

The notebook performs the following steps:

1. Import dependencies and setup parameters
2. Prepare the dataset
3. Predict using the original model
4. Transfer learning
5. Visualize the model output
6. Export the saved model


## Running the notebook

To run the notebook, follow the instructions in `setup.md`.

## References

Dataset citations:
```
@InProceedings{10.1007/978-3-540-76386-4_17,
    author="Wang, Liming
    and Shi, Jianbo
    and Song, Gang
    and Shen, I-fan",
    editor="Yagi, Yasushi
    and Kang, Sing Bing
    and Kweon, In So
    and Zha, Hongbin",
    title="Object Detection Combining Recognition and Segmentation",
    booktitle="Computer Vision -- ACCV 2007",
    year="2007",
    publisher="Springer Berlin Heidelberg",
    address="Berlin, Heidelberg",
    pages="189--199",
    abstract="We develop an object detection method combining top-down recognition with bottom-up image segmentation. There are two main steps in this method: a hypothesis generation step and a verification step. In the top-down hypothesis generation step, we design an improved Shape Context feature, which is more robust to object deformation and background clutter. The improved Shape Context is used to generate a set of hypotheses of object locations and figure-ground masks, which have high recall and low precision rate. In the verification step, we first compute a set of feasible segmentations that are consistent with top-down object hypotheses, then we propose a False Positive Pruning(FPP) procedure to prune out false positives. We exploit the fact that false positive regions typically do not align with any feasible image segmentation. Experiments show that this simple framework is capable of achieving both high recall and high precision with only a few positive training examples and that this method can be generalized to many object classes.",
    isbn="978-3-540-76386-4"
}

@INPROCEEDINGS{Geiger2012CVPR,
    author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2012}
}
```

