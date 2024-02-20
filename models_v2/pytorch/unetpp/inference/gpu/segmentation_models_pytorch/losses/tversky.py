#
#The MIT License
#
#Copyright (c) 2019, Pavel Iakubovskii
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.
#

from typing import List, Optional

import torch
from ._functional import soft_tversky_score
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from .dice import DiceLoss

__all__ = ["TverskyLoss"]


class TverskyLoss(DiceLoss):
    """Tversky loss for image segmentation task.
    Where FP and FN is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes input is raw logits
        smooth:
        ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``

    Return:
        loss: torch.Tensor

    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):

        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(mode, classes, log_loss, from_logits, smooth, ignore_index, eps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)
