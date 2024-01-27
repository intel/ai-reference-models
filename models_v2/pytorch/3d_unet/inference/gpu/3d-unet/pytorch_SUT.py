# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import json
import os
import sys
import time
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import torch
import intel_extension_for_pytorch
import torch.nn as nn
import torch.nn.functional as F

from torch.jit._recursive import wrap_cpp_module
from torch.quantization.quantize_jit import(
    convert_jit,
    prepare_jit
)
from torch.quantization import default_qconfig
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
from brats_QSL import get_brats_QSL

sys.path.insert(0, os.path.join(os.getcwd(), "nnUnet"))
from nnunet.training.model_restore import load_model_and_checkpoint_files

traced = False

class _3DUNET_PyTorch_SUT():
    def __init__(self, model_dir, preprocessed_data_dir, performance_count,
        run_fp16, run_int8, calib_iters, channels_last, asymm, uint8, profiling, folds,
        checkpoint_name):
        print("Loading PyTorch model...")
        model_path = os.path.join(model_dir, "plans.pkl")
        assert os.path.isfile(model_path), "Cannot find the model file {:}!".format(model_path)
        self.trainer, params = load_model_and_checkpoint_files(model_dir, folds, mixed_precision=False, checkpoint_name=checkpoint_name)
        self.trainer.load_checkpoint_ram(params[0], False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")
        self.warmup_iter = 3
        self.qsl = get_brats_QSL(preprocessed_data_dir, performance_count)
        self.run_fp16 = run_fp16
        self.run_int8 = run_int8
        self.calib_iters = calib_iters
        self.channels_last = channels_last
        self.profiling = profiling
        self.asymm = asymm
        self.uint8  = uint8

    def jit_calib(self, query_samples):
        img = self.qsl.get_features(query_samples[0].index)
        print("tracing sample id {:d} with shape ={:}".format(query_samples[0].index, img.shape))
        img = torch.from_numpy(img[np.newaxis, ...]).float()
        self.trainer.network.eval()
        img = img.to("xpu")
        model = self.trainer.network.to("xpu")
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last_3d)
            img = img.to(memory_format=torch.channels_last_3d)
        print("start jit tracing")
        model = model.to("cpu")
        img = img.to("cpu")
        modelJit = torch.jit.trace(model, img)
        print("cpu trace finished")
        modelJit = modelJit.to("xpu")
        modelJit = wrap_cpp_module(modelJit._c) #No folding here

        qscheme = torch.per_tensor_affine if self.asymm else torch.per_tensor_symmetric
        dtype = torch.quint8 if self.uint8 else torch.qint8

        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.observer.MinMaxObserver.with_args(
                qscheme=qscheme,
                reduce_range=False,
                dtype=dtype
            ),
            weight=torch.quantization.default_weight_observer
        )
        modelJit = prepare_jit(modelJit, {'': qconfig}, True)
        modelJit = modelJit.to('xpu')

        print("calib_iters:", self.calib_iters)
        for i in range(len(query_samples)):
            data = self.qsl.get_features(query_samples[i].index)
            print("calibrating through jit path, sample id {:d} with shape ={:}".format(query_samples[i].index, data.shape))
            image = torch.from_numpy(data[np.newaxis, ...]).float()
            calib = image.to("xpu")
            if self.channels_last:
                calib = calib.to(memory_format=torch.channels_last_3d)
            output = modelJit(calib)
            if i == self.calib_iters - 1:
                break
        print("start converting model")
        modelJit = convert_jit(modelJit)
        print("converting model ends")

        self.trainer.network = modelJit

        # setting global flag to avoid multiple times of tracing
        global traced
        traced = True

    def issue_queries(self, query_samples):
        if self.run_int8:
            if not traced:
                self.jit_calib(query_samples)
        total_time = 0
        counter = 0
        if self.run_int8:
            with torch.no_grad():
                for i in range(len(query_samples)):
                    print("evaluating", i, "th image")
                    data = self.qsl.get_features(query_samples[i].index)
                    print("Processing sample id {:d} with shape = {:}".format(query_samples[i].index, data.shape))

                    image = torch.from_numpy(data[np.newaxis,...]).float()
                    for layer in self.trainer.network.modules():
                        if isinstance(layer, torch.nn.InstanceNorm3d):
                            layer.float()
                    if self.channels_last:
                        image = image.to(memory_format=torch.channels_last_3d)
                        print("image to channels last")
                    try:
                        import memory_check
                        memory_check.display_mem("xpu:0")
                    except:
                        pass
                    if self.profiling and (i > self.warmup_iter):
                        with torch.autograd.profiler_legacy.profile(use_xpu=True, record_shapes=False) as prof:
                                image = image.to("xpu")
                                output = self.trainer.network(image)
                                output = output[0].cpu()
                                torch.xpu.synchronize()
                        # output profiling statiscs
                        profiling_path = os.path.abspath('../../') + '/report/'
                        torch.save(prof.key_averages().table(), profiling_path + '3dunet_inference_profiling.pt')
                        prof.export_chrome_trace(profiling_path + '3dunet_inference_profiling.json')
                        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                        print(prof.key_averages(group_by_input_shape=True).table())
                        print(prof.table(sort_by="id", row_limit=100000))
                        sys.exit(0)
                    else:
                        t0 = time.time()
                        image = image.to("xpu")
                        output = self.trainer.network(image)
                        output = output[0].cpu()
                        torch.xpu.synchronize()
                        t1 = time.time()
                        if i > self.warmup_iter:
                            total_time += (t1 - t0)
                            counter += 1
                    print("evaluated", i, "th image")
                    output = output.numpy().astype(np.float16)

                    transpose_forward = self.trainer.plans.get("transpose_forward")
                    transpose_backward = self.trainer.plans.get("transpose_backward")
                    assert transpose_forward == [0, 1, 2], "Unexpected transpose_forward {:}".format(transpose_forward)
                    assert transpose_backward == [0, 1, 2], "Unexpected transpose_backward {:}".format(transpose_backward)

                    response_array = array.array("B", output.tobytes())
                    bi = response_array.buffer_info()
                    response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                    lg.QuerySamplesComplete([response])
            if counter > 0:
                latency = total_time / counter
                throughput = counter / total_time
                print('3dunet_inf throughput: ', throughput, ' sample/s')
                print('3dunet_inf latency: ', latency, ' s')
            
        else:      
            model = self.trainer.network.to("xpu")
            model.eval()
            dtype = torch.float16 if self.run_fp16 else torch.float32
            model = torch.xpu.optimize(model=model, dtype=dtype, level="O1")
            for i in range(len(query_samples)):
                data = self.qsl.get_features(query_samples[i].index)
                print("calibrating through jit path, sample id {:d} with shape ={:}".format(query_samples[i].index, data.shape))
            use_autocast = False
            if self.run_fp16:
                print('using autocast')
                use_autocast = True

            autocast_dtype = torch.float32
            if self.run_fp16:
                autocast_dtype = torch.float16

            trace_input = torch.randn(1, 4, 224, 224, 160).to("xpu")
            with torch.no_grad():
                with torch.xpu.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    modelJit = torch.jit.trace(model, trace_input)
                    modelJit = torch.jit.freeze(modelJit)

            with torch.no_grad():
                for i in range(len(query_samples)):
                    print("evaluating", i, "th image")
                    data = self.qsl.get_features(query_samples[i].index)
                    print("Processing sample id {:d} with shape = {:}".format(query_samples[i].index, data.shape))

                    image = torch.from_numpy(data[np.newaxis,...]).float()
                    if self.run_fp16:
                        image = image.half()
                        self.trainer.network.half()
                    for layer in self.trainer.network.modules():
                        if isinstance(layer, torch.nn.InstanceNorm3d):
                            layer.float()
                    if self.channels_last:
                        image = image.to(memory_format=torch.channels_last_3d)
                        print("image to channels last")
                    try:
                        import memory_check
                        memory_check.display_mem("xpu:0")
                    except:
                        pass
                    if self.profiling and (i > self.warmup_iter):
                        with torch.autograd.profiler_legacy.profile(use_xpu=True, record_shapes=False) as prof:
                                image = image.to("xpu")
                                with torch.xpu.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                                    output = modelJit(image)
                                output = output[0].cpu()
                                torch.xpu.synchronize()
                        # output profiling statiscs
                        profiling_path = os.path.abspath('../../') + '/report/'
                        torch.save(prof.key_averages().table(), profiling_path + '3dunet_inference_profiling.pt')
                        prof.export_chrome_trace(profiling_path + '3dunet_inference_profiling.json')
                        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                        print(prof.key_averages(group_by_input_shape=True).table())
                        print(prof.table(sort_by="id", row_limit=100000))
                        sys.exit(0)
                    else:
                        t0 = time.time()
                        image = image.to("xpu")
                        with torch.xpu.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                            output = modelJit(image)
                        output = output[0].cpu()
                        torch.xpu.synchronize()
                        t1 = time.time()
                        if i > self.warmup_iter:
                            total_time += (t1 - t0)
                            counter += 1
                    print("evaluated", i, "th image")
                    output = output.numpy().astype(np.float16)

                    transpose_forward = self.trainer.plans.get("transpose_forward")
                    transpose_backward = self.trainer.plans.get("transpose_backward")
                    assert transpose_forward == [0, 1, 2], "Unexpected transpose_forward {:}".format(transpose_forward)
                    assert transpose_backward == [0, 1, 2], "Unexpected transpose_backward {:}".format(transpose_backward)

                    response_array = array.array("B", output.tobytes())
                    bi = response_array.buffer_info()
                    response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                    lg.QuerySamplesComplete([response])
            if counter > 0:
                latency = total_time / counter
                throughput = counter / total_time
                print('3dunet_inf throughput: ', throughput, ' sample/s')
                print('3dunet_inf latency: ', latency, ' s')

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass


def get_pytorch_sut(model_dir, preprocessed_data_dir, performance_count, run_fp16,
    run_int8, calib_iters, channels_last, asymm, uint8, profiling=False, folds=1,
    checkpoint_name="model_final_checkpoint"):
    return _3DUNET_PyTorch_SUT(model_dir, preprocessed_data_dir,
        performance_count, run_fp16, run_int8, calib_iters, channels_last, asymm, uint8,
        profiling, folds, checkpoint_name)
