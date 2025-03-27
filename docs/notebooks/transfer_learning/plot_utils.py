#!/usr/bin/env python
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

import os
import matplotlib.pyplot as plt


def plot_curves(history, checkpoint_dir):
    """
    Reads a pickle file and plots accuracy and loss curves

    :param history: Pickle file
    :return: None
    """
    if not history:
        raise FileNotFoundError("The pickle file {} does not exist".format(history))

    acc = history["acc"]
    val_acc = history["val_acc"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    plt.figure(figsize=(7, 7))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    if not os.path.exists(os.path.join(checkpoint_dir, "train_val_plot.png")):
        print("Saving plot in checkpoint_dir:", checkpoint_dir)
        plt.savefig(os.path.join(checkpoint_dir, "train_val_plot.png"))
