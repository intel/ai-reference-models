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
import os
import sys
import pickle
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np

# sample modules
import io_utils
from arguments_utils import args
import barrier_utils

from nnunet.training.model_restore import load_model_and_checkpoint_files

class brats2019Dataset(Dataset):
    """
    Needs a rootPath.
    Expects to find preprocessed_files.pkl file
    Expects to deal with preprocessed numpy files.
    """
    def __init__(self, preprocessed_data_dir, gt_info):
        self.source = []
        self.preprocessed_data_dir = preprocessed_data_dir
        self.gt_info = gt_info

        with open(os.path.join(self.preprocessed_data_dir, 'preprocessed_files.pkl'), "rb") as f:
            self.source = pickle.load(f)

    def __len__(self):
        return len(self.source)
    
    def __getitem__(self,idx):
        file_name = self.source[idx]
        with open(os.path.join(self.preprocessed_data_dir, "{:}.pkl".format(file_name)), "rb") as f:
            full_info = pickle.load(f)
            x = full_info[0]
            dct = full_info[1]
        x = torch.from_numpy(x).float()
        y = (file_name, dct)

        if self.gt_info is not None:
            self.gt_info[idx] = y
        return x, idx

class Reshape():
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return torch.reshape(img, (4, 224, 224, 160))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def load_model_from_nnunet():
    model_path = os.path.join(args.arch, "plans.pkl")
    assert os.path.isfile(model_path), "Cannot find the model file {:}!".format(model_path)
    trainer, params = load_model_and_checkpoint_files(args.arch, 1, checkpoint_name="model_final_checkpoint")
    trainer.load_checkpoint_ram(params[0], False)
    model = trainer.network

    # channels last
    # NOTE: this will be default memory format in future version of PyTorch/IPEX
    if args.channels_last:
        if args.xpu:
            io_utils.write_info('Model is converted to channels last')
            model = model.to(memory_format=torch.channels_last_3d)

    if args.gpu:
        trainer.network.to
        torch.cuda.set_device(args.device)
        model = model.cuda(args.device)
        io_utils.write_info('Model to cuda')
    elif args.xpu:
        torch.xpu.set_device(args.device)
        model = model.xpu(args.device)
        io_utils.write_info('Model to xpu')
    else:
        # do training or inference on CPU
        pass

    if not any([args.int8, args.uint8]): #TODO: Do we want this condition?
        io_utils.write_info('Doing torch xpu optimize for inference')
        model.eval()
        dtype = torch.float16 if args.fp16 else torch.float32
        dtype = torch.bfloat16 if args.bf16 else dtype
        if args.xpu:
            model = torch.xpu.optimize(model=model, dtype=dtype, level='O1')

    if args.gpu:
        cudnn.benchmark = True

    return model

def load_model():
    return load_model_from_nnunet()

def load_validation_dataset(batch_size, sample_number, data_dir=None, data_workers=4, pin_memory_device=None, gt_info=None):
    # Ensure we are loading even batches of data.
    # If not using dummy data this means the provided validation dataset must have at least batch_size worth of images.
    if sample_number < batch_size:
        io_utils.write_warning('Number of images to be loaded (dummy or real) is being increased to be a multiple of batch size.')
        io_utils.write_warning('From {0} images to {1} images.'.format(sample_number, batch_size))
        sample_number = batch_size
    if sample_number % batch_size != 0:
        io_utils.write_warning('Number of images to be loaded (dummy or real) is being truncated to be a multiple of batch size.')
        io_utils.write_warning('From {0} images to {1} images.'.format(sample_number, sample_number - sample_number % batch_size))
        sample_number -= sample_number % batch_size

    if data_dir == None:
        io_utils.write_info('Dummy data is used')
        validation_dataset_size = sample_number
        transform = transforms.Compose([
            transforms.ToTensor(),
            Reshape(),
        ])
        validation_dataset = datasets.FakeData(validation_dataset_size, (224 * 4, 224 * 160), 1000, transform=transform)
        validation_sampler = None

    else:
        if not os.path.exists(data_dir):
            io_utils.write_error('Dataset "{0}" to load does not exist!'.format(data_dir))
            sys.exit(1)

        # The following section is required to avoid multiple streams trying to simultaneously extract BraTS2019 Preprocessed dataset.
        if args.instance == 1:
            io_utils.write_info('Loading BraTS2019 preprocessed dataset')
            validation_dataset = brats2019Dataset(args.data, gt_info)
        barrier_utils.do_ipc_sync(args.barrier, 'load_3dunet', False)
        if args.instance != 1:
            io_utils.write_info('Loading BraTS2019 preprocessed dataset')
            validation_dataset = brats2019Dataset(args.data, gt_info)
        validation_sampler = [i for i in range(sample_number)]

    if pin_memory_device == None:
        return torch.utils.data.DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False,
            num_workers=data_workers, pin_memory=True, sampler=validation_sampler)
    else:
        return torch.utils.data.DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False,
            num_workers=data_workers, pin_memory=True, pin_memory_device=pin_memory_device, sampler=validation_sampler)

def validate_data_src(path_to_data, use_dummy):
    if use_dummy:
        return None

    if path_to_data == None:
        io_utils.write_error('A dataset must be provided through "--data [PATH]" argument or "--dummy" data must be enabled.')
        sys.exit(1)

    if not os.path.exists(path_to_data):
        io_utils.write_error('Dataset "{0}" does not exist!'.format(path_to_data))
        sys.exit(1)

    return path_to_data
