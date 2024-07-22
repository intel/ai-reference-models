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
import os
import io

# sample modules
import io_utils
from arguments_utils import args

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = pred.cpu()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}={avg' + self.fmt + '}{unit}'
        return fmtstr.format(**self.__dict__)

class progress_meter(object):
    def __init__(self, progress_max, meters, prefix=''):
        self.progress_max_fmtstr = self._get_progress_fmtstr(progress_max)
        self.meters = meters
        self.prefix = prefix

    def display(self, progress):
        entries = []
        entries += [str(meter) for meter in self.meters]
        io_utils.stdout_helper(self.prefix + self.progress_max_fmtstr.format(progress) + '\t' + ' | '.join(entries))

    def _get_progress_fmtstr(self, progress_max):
        if progress_max == None:
            return '[{}]'
        num_digits = len(str(progress_max // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(progress_max) + ']'

def log_raw_perf_to_file(file_name, *_args):
    file_path = os.sep.join([args.output_dir, file_name]) + '_{0}.csv'.format(args.instance)
    is_new_file = not os.path.exists(file_path)
    with io.open(file_path, 'a', encoding='utf-8') as file_obj:
        if is_new_file:
            file_obj.write(','.join(list(map(lambda p: '{0} ({1})'.format(p[0], p[1]) if p[1].strip() != '' else '{0}'.format(p[0]), _args))) + '\n')
        file_obj.write(','.join(list(map(lambda item: str(item[2]), _args))) + '\n')

