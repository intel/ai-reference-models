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
import torch.nn as nn
import timm
from timm.layers.norm_act import BatchNormAct2d
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# sample modules
import io_utils
from arguments_utils import args
import barrier_utils


# Optimizing Hugging Face FBNet model for IPEX through custom "monkey patch"
# - Timm's FBNet model uses BatchNormAct2d which has nn.functional.bach_norm, drop layer and and act(relu/identity) layer.
# - nn.functional.bach_norm uses pytorch's backend implementation instead of IPEX. This results in the layer to be executed on CPU.
# - This "monkey patch" replaces nn.functional.bach_norm with nn.BatchNorm2d which is supported by IPEX.
original_init = BatchNormAct2d.__init__

def custom_init(
    self,
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True,
    apply_act=True,
    act_layer=nn.ReLU,
    act_kwargs=None,
    inplace=True,
    drop_layer=None,
    device=None,
    dtype=None,
):
    original_init(
        self,
        num_features,
        eps,
        momentum,
        affine,
        track_running_stats,
        apply_act,
        act_layer,
        act_kwargs,
        inplace,
        drop_layer,
        device,
        dtype,
    )
    factory_kwargs = {"device": device, "dtype": dtype}
    self.batch_norm_layer = nn.BatchNorm2d(
        num_features=self.num_features,
        eps=self.eps,
        momentum=self.momentum,
        affine=self.affine,
        track_running_stats=self.track_running_stats,
        **factory_kwargs,
    )


def custom_forward(self, x):
    x = self.batch_norm_layer(x)
    x = self.drop(x)
    x = self.act(x)
    return x


def apply_monkey_patch():
    BatchNormAct2d.__init__ = custom_init
    BatchNormAct2d.forward = custom_forward

def create_optimized_fbnet_model(model_name, pretrained=True):
    if model_name == 'fbnetc_100':
        model_name += '.rmsp_in1k'
    pretrained_state_dict = None
    if pretrained: # Must be done before monkey patch
        pretrained_model = timm.create_model(model_name, pretrained=True)
        pretrained_state_dict = pretrained_model.state_dict()
    apply_monkey_patch()
    io_utils.write_warning('Applying custom BatchNormAct2d monkey patch')
    model = timm.create_model(model_name, pretrained=False)
    model_state_dict = model.state_dict()
    if pretrained:
        new_state_dict = {}
        for name, param in model_state_dict.items():
            if name in pretrained_state_dict and param.size() == pretrained_state_dict[name].size():
                parts = name.split(".")
                if any(part in ["bn1", "bn2", "bn3"] for part in parts) and all(
                    part != "batch_norm_layer" for part in parts
                ):
                    parts_without_param = parts[:-1]
                    name_without_param = ".".join(parts_without_param)
                    new_name = name_without_param + ".batch_norm_layer." + parts[-1]
                    new_state_dict[new_name] = pretrained_state_dict[name]
                else:
                    new_state_dict[name] = pretrained_state_dict[name]
        io_utils.write_warning('Applying modified weights to accommodate custom BatchNormAct2d monkey patch')
        model.load_state_dict(new_state_dict, strict=False)
    return model

# End of "monkey patch"

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

def load_model_from_huggingface():
    # Serialize model loading across multiple parallel instances to avoid errors on
    # processing the same weights file at the same time
    for index in range(args.total_instances):
        if index + 1 == args.instance:
            if args.pretrained:
                io_utils.write_info('Using pre-trained model "{0}"...'.format(args.arch))
                model = create_optimized_fbnet_model(args.arch, pretrained=True)
                io_utils.write_info('Pre-trained model "{0}" instantiated succesfully'.format(args.arch))
            else:
                io_utils.write_info('Using untrained model "{0}"...'.format(args.arch))
                model = create_optimized_fbnet_model(args.arch, pretrained=False)
                io_utils.write_info('Untrained model "{0}" instantiated succesfully'.format(args.arch))

        barrier_utils.do_ipc_sync(args.barrier, 'instance_{}_model_load'.format(index + 1), False)

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
        return load_model_from_huggingface()

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
