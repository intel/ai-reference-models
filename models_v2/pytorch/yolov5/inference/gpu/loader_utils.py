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
import platform
import torch
import torch.backends.cudnn as cudnn

# sample modules
import io_utils
from arguments_utils import args
import barrier_utils

try:
    from yolov5.utils.dataloaders import LoadImages
    from yolov5.models.common import DetectMultiBackend
except Exception as e:
    print('fatal: ' + str(e), file=sys.stderr)
    print('fatal: download yolov5 model sources to yolov5 folder', file=sys.stderr)
    print('fatal: or set PYTHONPATH to the location of yolov5 folder', file=sys.stderr)
    sys.exit(1)

def save_model_to_file(model):
    if args.save:
        if args.instance == 1: # Dont have multiple instances save to the same location
            io_utils.write_info('Saving model to "{0}"...'.format(args.save))
            if args.jit_trace or args.jit_script:
                torch.jit.save(model, args.save)
            elif any([args.int8, args.uint8]):
                torch.jit.save(model, args.save)
            else:
                torch.save(model, args.save)
            io_utils.write_info('Model saved succesfully')
        barrier_utils.do_ipc_sync(args.barrier, "save_model", args.terminate_if_sync_fail)
    else:
        io_utils.write_error('Save model path was not specified using "--save" argument!')
        sys.exit(1)

def load_model():
    model = DetectMultiBackend(args.weights, args.device, args.dnn, args.cfg, args.fp16)
    io_utils.write_info('Model yolov5m instantiated succesfully')

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

    if not any([args.int8, args.uint8]): #TODO: Do we want this condition?
        io_utils.write_info('Doing torch xpu optimize for inference')
        model.eval()
        dtype = torch.float16 if args.fp16 else torch.float32
        dtype = torch.bfloat16 if args.bf16 else dtype
        if args.xpu:
            import intel_extension_for_pytorch
            model = torch.xpu.optimize(model=model, dtype=dtype, level='O1')
            model = intel_extension_for_pytorch.optimize(model)

    if args.gpu:
        cudnn.benchmark = True

    return model

def dummy_dataset(validation_dataset_size, sample_width, sample_height):
    im = torch.torch.randn(validation_dataset_size, 3, sample_width, sample_height)
    path, s, cap = '', '', None
    
    return path, im, im, cap, s

def load_validation_dataset(batch_size, sample_number, sample_width, sample_height, model, data_dir=None, data_workers=4, pin_memory_device=None):
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

    if args.dummy:
        io_utils.write_info('Dummy data is used')
        validation_dataset_size = sample_number
        validation_dataset = [dummy_dataset(validation_dataset_size, sample_width, sample_height)]

    else:
        if not os.path.exists(data_dir):
            io_utils.write_error('Dataset "{0}" to load does not exist!'.format(data_dir))
            sys.exit(1)
       
        data_dir = str(data_dir)
        # The following section is required to avoid multiple streams trying to simultaneously extract Coco dataset.
        if args.instance == 1:
            io_utils.write_info('Loading Coco dataset')
            validation_dataset = LoadImages(data_dir, img_size=[sample_width, sample_height], stride=model.stride, auto=model.pt)
        
        barrier_utils.do_ipc_sync(args.barrier, 'load_coco', False)
        
        if args.instance != 1:
            io_utils.write_info('Loading Coco dataset')
            validation_dataset = LoadImages(data_dir, img_size=[sample_width, sample_height], stride=model.stride, auto=model.pt)
    
    return validation_dataset
