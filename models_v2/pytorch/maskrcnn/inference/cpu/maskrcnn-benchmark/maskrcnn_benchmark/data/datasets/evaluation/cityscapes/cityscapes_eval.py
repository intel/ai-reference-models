import logging
import tempfile
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
import os
import torch
from collections import OrderedDict
from tqdm import tqdm
from copy import deepcopy

import torch
import numpy as np

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from maskrcnn_benchmark.data.datasets.evaluation.cityscapes import eval_instances


from cityscapesscripts.helpers.csHelpers import writeDict2JSON, ensurePath


def do_cityscapes_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info(f"CityScapes evaluation on [{dataset}]:")
    # Set default args for evaluation
    args = deepcopy(eval_instances.defaultArgs)

    # Set output folder
    output_folder = os.path.join(output_folder, "evaluationResults")
    ensurePath(output_folder)

    # Set custom fields
    args.exportMatchFile = os.path.join(output_folder, "matches.json")
    args.exportBoxFile = os.path.join(output_folder, "boxResult.json")
    args.exportMaskFile = os.path.join(output_folder, "maskResult.json")
    args.instLabels = list(dataset.CLASSES)

    logger.info("Evaluation arguments")
    logger.info("%s" % args)
    logger.info("Matching GT instances with Predictions")
    if "bbox" in iou_types or "segm" in iou_types:
        # Match and compute IoU of mask and box in one iteration:
        matches = eval_instances.matchGtsWithPreds(dataset, predictions)
        writeDict2JSON(matches, args.exportMatchFile)
    else:
        NotImplementedError(f"IoU type not implemented {iou_types}")

    # printing
    strResults = ""
    if "bbox" in iou_types:
        # evaluate matches
        logger.info("Evaluating BBox matches")
        boxApScores = eval_instances.evaluateBoxMatches(matches, args)

        # averages
        logger.info("Average Box scores")
        boxAvgDict = eval_instances.computeAverages(boxApScores, args)

        # logging
        boxResDict = eval_instances.prepareJSONDataForResults(
            boxAvgDict, boxApScores, args
        )
        if args.JSONOutput:
            # create output folder if necessary
            path = os.path.dirname(args.exportBoxFile)
            ensurePath(path)
            # Write APs to JSON
            eval_instances.writeDict2JSON(boxResDict, args.exportBoxFile)
        strBoxResults = eval_instances.printResults(boxAvgDict, args)
        strResults += "\nBBox\n" + strBoxResults

    if "segm" in iou_types:
        # evaluate matches
        logger.info("Evaluating Mask matches")
        maskApScores = eval_instances.evaluateMaskMatches(matches, args)

        # averages
        logger.info("Average Mask scores")
        maskAvgDict = eval_instances.computeAverages(maskApScores, args)

        # logging
        maskResDict = eval_instances.prepareJSONDataForResults(
            maskAvgDict, maskApScores, args
        )
        if args.JSONOutput:
            # create output folder if necessary
            path = os.path.dirname(args.exportMaskFile)
            ensurePath(path)
            # Write APs to JSON
            eval_instances.writeDict2JSON(maskResDict, args.exportMaskFile)
        strMaskResults = eval_instances.printResults(maskAvgDict, args)
        strResults += "\nMask\n" + strMaskResults

    logger.info(strResults)
