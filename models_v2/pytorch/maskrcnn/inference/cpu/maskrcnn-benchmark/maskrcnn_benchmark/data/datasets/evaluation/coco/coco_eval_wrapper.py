# COCO style evaluation for custom datasets derived from AbstractDataset
# by botcs@github
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

import logging
import os
import json

from maskrcnn_benchmark.data.datasets.coco import COCODataset
from .coco_eval import do_coco_evaluation as orig_evaluation
from .abs_to_coco import convert_abstract_to_coco


def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Converting annotations to COCO format...")
    coco_annotation_dict = convert_abstract_to_coco(dataset)

    dataset_name = dataset.__class__.__name__
    coco_annotation_path = os.path.join(output_folder, dataset_name + ".json")
    logger.info("Saving annotations to %s" % coco_annotation_path)
    with open(coco_annotation_path, "w") as f:
        json.dump(coco_annotation_dict, f, indent=2)

    logger.info("Loading annotations as COCODataset")
    coco_dataset = COCODataset(
        ann_file=coco_annotation_path,
        root="",
        remove_images_without_annotations=False,
        transforms=None,  # transformations should be already saved to the json
    )

    return orig_evaluation(
        dataset=coco_dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results,
    )
