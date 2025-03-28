# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
# MIT License
# 
# Copyright (c) 2018 Facebook
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
Simple dataset class that wraps a list of path names
"""

from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList


class ListDataset(object):
    def __init__(self, image_lists, transforms=None):
        self.image_lists = image_lists
        self.transforms = transforms

    def __getitem__(self, item):
        img = Image.open(self.image_lists[item]).convert("RGB")

        # dummy target
        w, h = img.size
        target = BoxList([[0, 0, w, h]], img.size, mode="xyxy")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        pass
