#
# Copyright (c) 2023 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import time
import torch
import intel_extension_for_pytorch as ipex
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def write_profile_info(prof, bs):
    title = "hifigan_inference_"
    title += "fp16" + "_"
    title += "bs" + str(bs) + "_"
    title += "xpu_"

    profiling_path = os.getenv('PROFILE_PATH')
    if not profiling_path:
        profiling_path = ''
    sort_str = "self_xpu_time_total"
    torch.save(prof.key_averages().table(sort_by=sort_str), profiling_path + title + 'profiling.pt')
    torch.save(prof.key_averages(group_by_input_shape=True).table(), profiling_path + title + 'profiling_detailed.pt')
    prof.export_chrome_trace(profiling_path + title + 'profiling.json')
    print(prof.key_averages().table(sort_by=sort_str))
    print(prof.key_averages().table(sort_by=sort_str), file=open(profiling_path + title + '.breakdown', 'w'))
    print(prof.key_averages(group_by_input_shape=True).table())

def print_perf_info(total_time, counter, batch_size):
    latency = total_time / counter
    throughput = batch_size / latency
    print("----------------")
    print("avg latency: ", latency)
    print("throughput: ", throughput)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(torch.device('cpu'))

    state_dict_g = load_checkpoint(a.checkpoint_file, torch.device('cpu'))
    generator.load_state_dict(state_dict_g['generator'])
    generator = generator.to(device)
    datatype = torch.float16

    filelist = os.listdir(a.input_wavs_dir)
    os.makedirs(a.output_dir, exist_ok=True)
    dummy = True if a.dummy else False

    batch_size = a.batch_size if dummy else 1
    if batch_size <= 0:
      batch_size = 1
    iteration = a.iter
    if iteration <=0:
        iteration = 500
    print("total iteration:", iteration)
    print("use dummy data:", dummy)
    print("batch_size", batch_size)
    print("device:", device)
    print("jit:", a.torchjit)

    if device == torch.device('xpu'):
      generator.eval()
      generator.remove_weight_norm()
      ipex.enable_auto_channels_last()
      generator = torch.xpu.optimize(model=generator, dtype=datatype)

    if a.dynamic_shape == False:
        filelist = ['LJ038-0254.wav'] * a.batch_size*(iteration+20)

    if a.torchjit and device == torch.device('xpu'):
      for i, filname in enumerate(filelist):
          if i==0:
               wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
               wav = wav / MAX_WAV_VALUE
               wav = torch.FloatTensor(wav)
               x = get_mel(wav.unsqueeze(0))
               if dummy:
                   x = torch.randn([batch_size, x.size(1), x.size(2)])
               x = x.to(device)
               with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                   generator = torch.jit.trace(generator, x, check_trace=False)
                   generator = torch.jit.freeze(generator)
               break

    generator = generator.to(device)
    total_time = 0.0
    counter = 0

    with torch.no_grad():
        for i, filname in enumerate(filelist):
            if i < iteration:
                print("--------iteration:", i, "filename:", filname)
                profiling_enabled = True if (i == 45) else False
                wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
                wav = wav / MAX_WAV_VALUE
                wav = torch.FloatTensor(wav)
                x = get_mel(wav.unsqueeze(0))
                if(dummy):
                    x = torch.randn([batch_size, x.size(1), x.size(2)])

                print("size:", x.shape)

                with torch.autograd.profiler_legacy.profile(enabled=profiling_enabled, use_xpu=True, record_shapes=False) as prof:
                    start_time = time.time()
                    x = x.to(device)
                    with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                        y_g_hat = generator(x)
                        audio = y_g_hat.squeeze()
                        audio = audio * MAX_WAV_VALUE
                        audio = audio.cpu()

                    if device=="xpu":
                        torch.xpu.synchronize()
                    end_time = time.time()

                audio = audio.numpy().astype('int16')

                if i > 5:
                  total_time += (end_time - start_time)
                  counter += 1

                if profiling_enabled:
                    print(prof.key_averages().table(sort_by="xpu_time_total"))
                    write_profile_info(prof, batch_size)

                if not dummy:
                    output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
                    write(output_file, h.sampling_rate, audio)
                    print(output_file)

                if i == iteration - 1:
                  print_perf_info(total_time, counter, batch_size)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--iter', type=int, default=500, help="iteration num")
    parser.add_argument('--dynamic_shape', default=False, action='store_true', help='use dynamic shape')
    parser.add_argument('--dummy', action='store_true', help="use dummy input or real dataset")
    parser.add_argument('--torchjit', action='store_true', help="use jit or not")
    parser.add_argument('--no_fix_shape', default=False, action='store_true', help="Use not fix shape data")
    parser.add_argument('--batch_size', type=int, default=8, help="set batchsize for dummy input data")
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    global device
    if torch.xpu.is_available():
        torch.xpu.manual_seed(h.seed)
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

