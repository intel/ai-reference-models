from .coco_eval import do_coco_evaluation as do_orig_coco_evaluation
from .coco_eval_wrapper import do_coco_evaluation as do_wrapped_coco_evaluation
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
from maskrcnn_benchmark.data.datasets import AbstractDataset, COCODataset


def coco_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    if isinstance(dataset, COCODataset):
        return do_orig_coco_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    elif isinstance(dataset, AbstractDataset):
        return do_wrapped_coco_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    else:
        raise NotImplementedError(
            (
                "Ground truth dataset is not a COCODataset, "
                "nor it is derived from AbstractDataset: type(dataset)="
                "%s" % type(dataset)
            )
        )
