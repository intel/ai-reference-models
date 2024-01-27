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

import torch
from pydoc import locate


# Dictionary of Torchvision image classification models
torchvision_model_map = {
    "resnet18": {
        "classifier": "fc"
    },
    "resnet50": {
        "classifier": "fc"
    },
    "efficientnet_b0": {
        "classifier": ["classifier", 1]
    },
    "efficientnet_b1": {
        "classifier": ["classifier", 1]
    },
    "efficientnet_b2": {
        "classifier": ["classifier", 1]
    },
    "densenet121": {
        "classifier": "classifier"
    },
    "densenet161": {
        "classifier": "classifier"
    },
    "googlenet": {
        "classifier": "fc"
    },
    "shufflenet_v2_x1_0": {
        "classifier": "fc"
    },
    "mobilenet_v2": {
        "classifier": ["classifier", 1]
    }
}


def get_retrainable_model(model_name, num_classes, do_fine_tuning=False):
    # Load an image classification model pretrained on ImageNet
    pretrained_model_class = locate('torchvision.models.{}'.format(model_name))
    classifier_layer = torchvision_model_map[model_name]['classifier']

    model = pretrained_model_class(pretrained=True)

    if not do_fine_tuning:
        for param in model.parameters():
            param.requires_grad = False

    if isinstance(classifier_layer, list):
        classifier = getattr(model, classifier_layer[0])[classifier_layer[1]]
        num_features = classifier.in_features
        model.classifier[classifier_layer[1]] = torch.nn.Linear(num_features, num_classes)
    else:
        classifier = getattr(model, classifier_layer)
        num_features = classifier.in_features
        setattr(model, classifier_layer, torch.nn.Linear(num_features, num_classes))

    return model
