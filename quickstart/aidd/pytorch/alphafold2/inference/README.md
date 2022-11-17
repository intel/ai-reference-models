# AlphaFold2 inference optimized on Intel-Architecture

## Description

<u>**Declaration 1**</u>
This repository contains an inference pipeline of AlphaFold2 with a *bona fide* translation from *Haiku/JAX* (https://github.com/deepmind/alphafold) to PyTorch.
Any publication that discloses findings arising from using this source code or the model parameters should [cite](#citing-this-work) the [AlphaFold paper](https://doi.org/10.1038/s41586-021-03819-2). Please also refer to the [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf) for a detailed description of the method.

<u>**Declaration 2**</u>
The setup procedures were modified from the two repos:
   https://github.com/kalininalab/alphafold_non_docker
   https://github.com/deepmind/alphafold
with only some exceptions. I will label the difference for highlight.

<u>**Declaration 3**</u>
This repo is independently implemented, and is different from a previously unofficial version (https://github.com/lucidrains/alphafold2).
No one is better than the other, and the differences are in 3 points:
(1) this repo is major in acceleration of inference, in compatible to weights released from DeepMind;
(2) this repo delivers a reliable pipeline accelerated on Intel® Xeon and Intel® Optane® PMem by Intel® oneAPI, which are alternative ways to deploy the model.
(3) this repo places CPU as its primary computation resource for acceleration, which may not provide an optimal speed on GPU.

## Bare Metal

### General setup
<u>**Primary solution for setup of alphafold2 environment optimized on Intel-Architecture 3**</u>

1. install anaconda;

1. create conda environment:
   ```bash
   conda create -n iaf2 python=3.9.7
   conda activate iaf2
   ```

### Model Specific Setup

1. initialize by running setup_env.sh:
   ```bash
   bash setup_env.sh \
     <root_home> \
     <refdata_dir> \
     <conda_env_name> \
     <experiment_name> \
     <model_name>
   ```

## Quick Start Scripts

There is already an example input "sample.fa" copied into your subfolder <root_home>/samples/ .

1. run squential preprocessing (MSA and template search) on samples in $root_home/samples

   ```bash
   bash online_preproc_baremetal.sh \
     <root_home> \ # root of all intermediate data
     <data_dir> \ # root of reference dataset that AlphaFold2 needs
     <input_dir> \ # Example: <path_to_root_home>/samples
     <out_dir> ## Example: <path_to_root_home>/experiments/<experiment_name (created in setup)
   ```
   
   <input_dir>=<root_home>/samples
   <out_dir>=<root_home>/experiments/<customized_subfolder>
   intermediates data can be seen under $root_home/experiments/<sample-name>/intermediates and $root_home/experiments/<sample-name>/msas
   these datafiles will be used as input of modelinfer
   
1. run sequential model inference to predict unrelaxed structures from MSA and template results

   ```bash
   bash online_inference_baremetal.sh \
     <path_to_condaenv> \ # the path to your conda virtual environment, e.g. ~/anaconda3/envs/<env_name>
     <root_home> \ # root of all intermediate data
     <root_data> \ # root of reference dataset that AlphaFold2 needs
     <input_dir> \ # Example: <path_to_root_home>/samples
     <out_dir> \ # Example: <path_to_root_home>/experiments/<experiment_name (created in setup)
     <model_name> # model_1
   ```

   <input_dir>=<root_home>/samples
   <out_dir>=<root_home>/experiments/<customized_subfolder>
   unrelaxed data can be seen under $root_home/experiments/<sample-name>
   now you can visualize the PDB files
   
## Run the model

1. run batch preprocessing (MSA and template search) on samples in $root_home/samples

   ```bash
   bash batch_preproc_baremetal.sh \
     <root_home> \ # root of all intermediate data
     <data_dir> \ # root of reference dataset that AlphaFold2 needs
     <input_dir> \ # Example: <path_to_root_home>/samples
     <out_dir> # Example: <path_to_root_home>/experiments/<experiment_name (created in setup)
   ```
   
   intermediates data can be seen under $root_home/experiments/<sample-name>/intermediates and $root_home/experiments/<sample-name>/msas
   these datafiles will be used as input of modelinfer
   
1. run batch model inference to predict unrelaxed structures from MSA and template results

   ```bash
   bash batch_inference_baremetal.sh \
     <path_to_condaenv> \ # the path to your conda virtual environment, e.g. ~/anaconda3/envs/<env_name>
     <root_home> \ # root of all intermediate data
     <root_data> \ # root of reference dataset that AlphaFold2 needs
     <input_dir> \ # Example: <path_to_root_home>/samples
     <out_dir> \ # Example: <path_to_root_home>/experiments/<experiment_name (created in setup)
     <model_name> # model_1
   ```

   unrelaxed data can be seen under $root_home/experiments/<sample-name>
   now you can visualize the PDB files

1. Notices:
   
   the optimal parallel thread number depends on the max memory size
   if you have PMem installed on your system, please use 1 physical core per thread
   if you only have DRAM memory of GB level, please estimate memory peak before use


## datasets
DeepMind provided scripts ... to download and cook all datasets as reference:
Default usage:
  <INSTALL_ROOT>/models/aidd/pytorch/alphafold2/inference/alphafold/scripts/download_all_data.sh <DOWNLOAD_DIR>
to use reduced_dbs:
  <INSTALL_ROOT>/models/aidd/pytorch/alphafold2/inference/alphafold/scripts/download_all_data.sh <DOWNLOAD_DIR> reduced_dbs
please find extra scripts to download specific datasets
  <INSTALL_ROOT>/models/aidd/pytorch/alphafold2/inference/alphafold/scripts/*

## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{AlphaFold2021,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  journal = {Nature},
  title   = {Highly accurate protein structure prediction with {AlphaFold}},
  year    = {2021},
  doi     = {10.1038/s41586-021-03819-2},
  note    = {(Accelerated article preview)},
}
```

## License and Disclaimer

Copyright (c) 2021 DeepMind Technologies Limited.
Copyright (c) 2022 Intel Corporation Limited.

### AlphaFold2 Code License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

### Model Parameters License

The AlphaFold parameters are made available for non-commercial use only, under
the terms of the Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0) license. You can find details at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

