# Transfer Learning for Image Classification with TF Hub

This notebook uses transfer learning with multiple [TF Hub](https://tfhub.dev) image classifiers,
[TF datasets](https://www.tensorflow.org/datasets/), and custom image datasets.

The notebook performs the following steps:
1. Import dependencies and setup parameters
1. Prepare the dataset
1. Predict using the original model
1. Transfer Learning
1. Evaluate the model
1. Export the saved model

## Running the notebook

To run the notebook, follow the instructions in `setup.md`.

## References

Dataset citations
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
