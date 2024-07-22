# MIT License

# Copyright (c) 2022 Lingtong Kong
# Copyright (c) 2023-2024 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This is a simplified implementation of the Vimeo90K Test Dataset loader
# and quality check functions derived from the original IFRNet repository.
# Following are the relevant source files (License in header)
# https://github.com/ltkong218/IFRNet/blob/b117bcafcf074b2de756b882f8a6ca02c3169bfe/datasets.py
# https://github.com/ltkong218/IFRNet/blob/b117bcafcf074b2de756b882f8a6ca02c3169bfe/metric.py

# IFRNet Citation (https://github.com/ltkong218/IFRNet/blob/b117bcafcf074b2de756b882f8a6ca02c3169bfe/README.md)
# @InProceedings{Kong_2022_CVPR,
#                  author = {Kong, Lingtong and Jiang, Boyuan and Luo, Donghao and Chu, Wenqing and Huang, Xiaoming and Tai, Ying and Wang, Chengjie and Yang, Jie},
#                  title = {IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation},
#                  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#                  year = {2022}
# }
#
# Dataset Citation: (From readme.txt within the Vimeo_90K test dataset)
# Archive capturing dataset: http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
# Webpage with info about the dataset: http://toflow.csail.mit.edu/
#
# @article{xue17toflow,
#          author = {Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
#          title = {Video Enhancement with Task-Oriented Flow},
#          journal = {arXiv},
#          year = {2017}
# }

import torch
from torch.utils.data import Dataset
import os
from imageio import imread

def calculate_psnr(img1, img2):
    psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
    return psnr

def quality_check(img, res, threshold=25):
    psnr = calculate_psnr(img, res)
    return True if psnr >= threshold else False

class Vimeo90K_Test_Dataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        with open(os.path.join(dataset_dir, 'tri_testlist.txt'), 'r') as f:
            for line in f:
                name = line.strip()
                if(len(name) <= 1):
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'input', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'target', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'input', name, 'im3.png'))                                

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        #Read png images
        img0 = imread(self.img0_list[idx])
        imgt = imread(self.imgt_list[idx])
        img1 = imread(self.img1_list[idx])
        # Convert to CHW
        img0 = torch.from_numpy(img0.transpose((2, 0, 1)))
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)))
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)))
        return img0, imgt, img1

