# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.utils.tensorboard as tb


class DummyLogger:
    def log_scalar(*args, **kwargs):
        pass

    def log_params(*args, **kwargs):
        pass

    def log_grad(*args, **kwargs):
        pass

    def train_end(*args, **kwargs):
        pass


class TensorBoardLogger(DummyLogger):
    def __init__(self, path, model, histogram=False):
        self.writer = tb.SummaryWriter(log_dir=str(path))
        self.model = model
        self.histogram = histogram

    def log_scalar(self, name, value, step, stage="train"):
        self.writer.add_scalar(f"{stage}/{name}", value, global_step=step)

    def log_grad(self, step):
        if not self.histogram:
            return
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    name.replace(".", "/"), param.grad, global_step=step
                )

    def log_params(self, step):
        if not self.histogram:
            return
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name.replace(".", "/"), param, global_step=step)

    def train_end(self):
        self.writer.close()
