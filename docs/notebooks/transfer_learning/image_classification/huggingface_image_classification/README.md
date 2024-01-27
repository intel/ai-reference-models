# Transfer Learning for Image Classification using HuggingFace

This notebook uses [ViT](https://huggingface.co/google/vit-base-patch16-224-in21k) classifier model from 🤗 model hub that was originally trained using [ImageNet](https://image-net.org) and does transfer learning with [Food101](https://huggingface.co/datasets/food101) dataset from 🤗 Datasets.

The notebook performs the following steps:
1. Import dependencies and setup parameters
2. Load the Food101 dataset
3. Preprocess the dataset
4. Transfer Learning (Pytorch/TensorFlow)
5. Predict on test subset

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
```
