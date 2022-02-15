#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import torch
import torchvision
import torchvision.models.detection as detection


# Dictionary of torchvision object detection models
torchvision_model_map = {
    "fasterrcnn_resnet50_fpn": {
        "predictor_model": "faster_rcnn.FastRCNNPredictor"
    },
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "predictor_model": "faster_rcnn.FastRCNNPredictor"
    },
    "fasterrcnn_mobilenet_v3_large_320_fpn": {
        "predictor_model": "faster_rcnn.FastRCNNPredictor"
    },
    "retinanet_resnet50_fpn": {
        "predictor_model": "retinanet.RetinaNetClassificationHead"
    }
}

def get_retrainable_model(model_name, num_classes, pretrained_model_class, predictor_class):
    # Load an object detection model pre-trained on COCO
    model = pretrained_model_class(pretrained=True)
    
    if 'fasterrcnn' in model_name:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = predictor_class(in_features, num_classes)
        
    elif 'retinanet' in model_name:
        # To keep the pretrained weights, do not overwrite the classification_head
        in_features = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = num_classes
        cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
        model.head.classification_head.cls_logits = cls_logits
        
    return model

