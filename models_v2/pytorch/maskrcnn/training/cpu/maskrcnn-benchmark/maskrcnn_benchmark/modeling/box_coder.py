# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
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

import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        # boxes = boxes.to(rel_codes.dtype)

        # TO_REMOVE = 1  # TODO remove
        # widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        # heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        # ctr_x = boxes[:, 0] + 0.5 * widths
        # ctr_y = boxes[:, 1] + 0.5 * heights

        # wx, wy, ww, wh = self.weights
        # dx = rel_codes[:, 0::4] / wx
        # dy = rel_codes[:, 1::4] / wy
        # dw = rel_codes[:, 2::4] / ww
        # dh = rel_codes[:, 3::4] / wh

        # # Prevent sending too large values into torch.exp()
        # dw = torch.clamp(dw, max=self.bbox_xform_clip)
        # dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        # pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        # pred_w = torch.exp(dw) * widths[:, None]
        # pred_h = torch.exp(dh) * heights[:, None]

        # pred_boxes = torch.zeros_like(rel_codes)
        # # x1
        # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # # y1
        # pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        rel_codes = rel_codes.to(torch.float32)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx + 0.5
        dy = rel_codes[:, 1::4] / wy + 0.5
        exp_dw = 0.5 * torch.exp(torch.clamp(rel_codes[:, 2::4] / ww, max=self.bbox_xform_clip))
        exp_dh = 0.5 * torch.exp(torch.clamp(rel_codes[:, 3::4] / wh, max=self.bbox_xform_clip))

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = (dx - exp_dw) * widths[:, None] + (boxes[:, 0])[:, None]
        # y1
        pred_boxes[:, 1::4] = (dy - exp_dh) * heights[:, None] + (boxes[:, 1])[:, None]
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = (dx + exp_dw) * widths[:, None] + (boxes[:, 0])[:, None] - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = (dy + exp_dh) * heights[:, None] + (boxes[:, 1])[:, None] - 1

        return pred_boxes
