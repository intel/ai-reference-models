# Transfer Learning for Image Classification using PyTorch

This notebook uses image classification models from Torchvision that were originally trained 
using ImageNet and does transfer learning with the Food101 dataset, a flowers dataset, or
a custom image dataset.

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
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}

@ONLINE {tfflowers,
author = "The TensorFlow Team",
title = "Flowers",
month = "jan",
year = "2019",
url = "http://download.tensorflow.org/example_images/flower_photos.tgz" }
```
