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
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# sample modules
import io_utils
from arguments_utils import args
import barrier_utils

def save_model_to_file(model):
    if args.save:
        if args.instance == 1: # Dont have multiple instances save to the same location
            io_utils.write_info('Saving model to "{0}"...'.format(args.save))
            if args.jit_trace or args.jit_script:
                torch.jit.save(model, args.save)
            else:
                torch.save(model, args.save)
            io_utils.write_info('Model saved succesfully')
        barrier_utils.do_ipc_sync(args.barrier, "save_model", args.terminate_if_sync_fail)
    else:
        io_utils.write_error('Save model path was not specified using "--save" argument!')
        sys.exit(1)

def load_model_from_file():
    if os.path.isfile(args.load):
        io_utils.write_info('Loading model from "{0}"...'.format(args.load))
        if args.jit_trace or args.jit_script:
            model = torch.jit.load(args.load)
        else:
            model = torch.load(args.load)
        io_utils.write_info('Model loaded succesfully')
    else:
        io_utils.write_error('No saved model found at "{0}"'.format(args.load))
        sys.exit(1)

    return model

def load_model_from_torchvision():
    if args.pretrained:
        io_utils.write_info('Using pre-trained model "{0}"...'.format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        io_utils.write_info('Pre-trained model "{0}" instantiated succesfully'.format(args.arch))
    else:
        io_utils.write_info('Using untrained model "{0}"...'.format(args.arch))
        model = models.__dict__[args.arch]()
        io_utils.write_info('Untrained model "{0}" instantiated succesfully'.format(args.arch))

    # channels last
    # NOTE: this will be default memory format in future version of PyTorch/IPEX
    if args.channels_last:
        io_utils.write_info('Model is converted to channels last')
        model = model.to(memory_format=torch.channels_last)

    if args.gpu:
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

    model.eval()

    if args.xpu and args.ipex:
        io_utils.write_info('Doing torch xpu optimize for inference')
        dtype = torch.float16 if args.fp16 else torch.float32
        dtype = torch.bfloat16 if args.bf16 else dtype
        model = torch.xpu.optimize(model=model, dtype=dtype, level='O1')

    if args.gpu:
        cudnn.benchmark = True

    return model

def load_model():
    if args.load:
        return load_model_from_file()
    else:
        return load_model_from_torchvision()

def load_validation_dataset(batch_size, sample_number, sample_width, sample_height, data_dir=None, data_workers=4, pin_memory_device=None):
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
        validation_dataset = datasets.FakeData(validation_dataset_size, (3, sample_width, sample_height), 1000, transforms.ToTensor())
        validation_sampler = None

    else:
        if not os.path.exists(data_dir):
            io_utils.write_error('Dataset "{0}" to load does not exist!'.format(data_dir))
            sys.exit(1)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((sample_width, sample_height)),
            transforms.CenterCrop((sample_width, sample_height)),
            transforms.ToTensor(),
            normalize,
        ])

        # The following section is required to avoid multiple streams trying to simultaneously extract ImageNet dataset.
        if args.instance == 1:
            io_utils.write_info('Loading ImageNet dataset')
            validation_dataset = datasets.ImageNet(
                data_dir,
                split='val',
                transform=transform
            )
        barrier_utils.do_ipc_sync(args.barrier, 'load_imagenet', False)
        if args.instance != 1:
            io_utils.write_info('Loading ImageNet dataset')
            validation_dataset = datasets.ImageNet(
                data_dir,
                split='val',
                transform=transform
            )
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
