# MIT License
#
# Copyright (c) 2019 Hao Gao
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
#

from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose(
            [
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(self.mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
                lambda img, boxes=None, labels=None: (img / std, boxes, labels),
                ToTensor(),
            ]
        )

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose(
            [
                ToPercentCoords(),
                Resize(size),
                SubtractMeans(mean),
                lambda img, boxes=None, labels=None: (img / std, boxes, labels),
                ToTensor(),
            ]
        )

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose(
            [
                Resize(size),
                SubtractMeans(mean),
                lambda img, boxes=None, labels=None: (img / std, boxes, labels),
                ToTensor(),
            ]
        )

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
