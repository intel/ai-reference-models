# Copyright (c) 2023-2024 Intel Corporation
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

# system modules
import torch
import torch.distributed as dist
import numpy as np

# sample modules
import io_utils

from nnunet.evaluation.region_based_evaluation import evaluate_case, get_brats_regions
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
import os
import uuid

def accuracy(args, outputs, target, overall, whole, core, enhancing, gt_data):
    for i in range(0, args.batch_size):
        output = outputs.data[i]
        output = output.cpu()
        output = output.to(dtype=torch.float16)

        # Find corresponding GT file
        gt_file = os.path.join(args.label_data_dir, f'{gt_data[target.data[i].item()][0]}.nii.gz')
        if not os.path.exists(gt_file):
            io_utils.write_warning('Unable to find Corresponding GT file for frame')
            continue

        result = output.numpy().astype(np.float16)

        # For Accuracy Check Need to save output prediction to .nii.gz
        padded_shape = [224, 224, 160]
        raw_shape = list(gt_data[target.data[i].item()][1]["size_after_cropping"])
        # Remove the padded part
        pad_before = [(p - r) // 2 for p, r in zip(padded_shape, raw_shape)]
        pad_after = [-(p - r - b) for p, r, b in zip(padded_shape, raw_shape, pad_before)]
        softmax_mean =  result[:, pad_before[0]:pad_after[0], pad_before[1]:pad_after[1], pad_before[2]:pad_after[2]]
        output_filename = str(uuid.uuid4()) + ".nii.gz"
        dct = gt_data[target.data[i].item()][1]
        save_segmentation_nifti_from_softmax(softmax_mean, output_filename, dct, 3, None, None, None,
                                             None, None, None, 0)
        # Run Accuracy on Output
        regions = get_brats_regions()
        accuracy_results = evaluate_case(os.path.abspath(output_filename), os.path.abspath(gt_file), list(regions.values()))

        # Clean Up and report Accuracy
        os.remove(output_filename)

        whole.update(np.nansum(accuracy_results[0]) * 100)
        core.update(np.nansum(accuracy_results[1]) * 100)
        enhancing.update(np.nansum(accuracy_results[2]) * 100)
        overall.update(np.nansum(accuracy_results[0]) * 100)
        overall.update(np.nansum(accuracy_results[1]) * 100)
        overall.update(np.nansum(accuracy_results[2]) * 100)

class average_meter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', unit=''):
        self.name = name
        self.fmt = fmt
        self.unit = unit
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}={avg' + self.fmt + '}{unit}'
        return fmtstr.format(**self.__dict__)

class progress_meter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = []
        entries += [str(meter) for meter in self.meters]
        io_utils.stdout_helper(self.prefix + self.batch_fmtstr.format(batch) + '\t' + ' | '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
