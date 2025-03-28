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
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
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

import argparse
import itertools
from typing import List
from tqdm import tqdm
import math
import toml
from dataset import AudioToTextDataLayer
from helpers import process_evaluation_batch, process_evaluation_epoch, Optimization, add_blank_label, AmpOptimizations, print_dict
from decoders import RNNTGreedyDecoder
from model_rnnt import RNNT
from preprocessing import AudioPreprocessing
from parts.features import audio_from_file
import torch
#import apex
#from apex import amp
import random
import numpy as np
import pickle
import time
import os

import torchvision
import intel_extension_for_pytorch

def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--local_rank", default=None, type=int, help='local rank')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='data batch size')
    parser.add_argument("--steps", default=None, help='if not specified do evaluation on full dataset. otherwise only evaluates the specified number of iterations for each worker', type=int)
    parser.add_argument("--model_toml", type=str, help='relative model configuration path given dataset folder')
    parser.add_argument("--dataset_dir", type=str, help='absolute path to dataset folder')
    parser.add_argument("--val_manifest", type=str, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--ckpt", default=None, type=str, required=True, help='path to model checkpoint')
    parser.add_argument("--max_duration", default=None, type=float, help='maximum duration of sequences. if None uses attribute from model configuration file')
    parser.add_argument("--pad_to", default=None, type=int, help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    parser.add_argument("--fp16", action='store_false', help='use half precision', default=False)
    parser.add_argument("--cudnn_benchmark", action='store_true', help="enable cudnn benchmark")
    parser.add_argument("--save_prediction", type=str, default=None, help="if specified saves predictions in text form at this location")
    parser.add_argument("--logits_save_to", default=None, type=str, help="if specified will save logits to path")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--wav", type=str, help='absolute path to .wav file (16KHz)')
    parser.add_argument("--xpu", action='store_true', help='use xpu', default=False)
    parser.add_argument("--int8", default=0, type=int, help='Datatype used: int8')
    parser.add_argument('--bf16', default=0, type=int, help='Datatype used: bf16')
    parser.add_argument("--warm_up", help='warm up steps, will only measure the performance from step=warm_up to step=(steps-warm_up)', type=int, default=3)
    # parser.add_argument('--channels_last', default=0, type=int, help='Dataformat used: channel_last(plain NHWC)')
    #store_true True 
    return parser.parse_args()

def eval(
        data_layer,
        audio_processor,
        encoderdecoder,
        greedy_decoder,
        labels,
        multi_gpu,
        args):
    """performs inference / evaluation
    Args:
        data_layer: data layer object that holds data loader
        audio_processor: data processing module
        encoderdecoder: acoustic model
        greedy_decoder: greedy decoder
        labels: list of labels as output vocabulary
        multi_gpu: true if using multiple gpus
        args: script input arguments
    """
    logits_save_to=args.logits_save_to
    encoderdecoder.eval()
    with torch.no_grad():
        _global_var_dict = {
            'predictions': [],
            'transcripts': [],
            'logits' : [],
        }


        
        if args.wav:
            features, p_length_e = audio_processor(audio_from_file(args.wav))
            #torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_log_probs_e = encoderdecoder(features)
            #torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_predictions_e = greedy_decoder(log_probs=t_log_probs_e)
            hypotheses = __ctc_decoder_predictions_tensor(t_predictions_e, labels=labels)
            print("INFERENCE TIME\t\t: {} ms".format((t1-t0)*1000.0))
            print("TRANSCRIPT\t\t:", hypotheses[0])
            return
        print("\nstart measure performance, measure steps = ", args.steps)
        total_time = 0

        for it, data in enumerate(tqdm(data_layer.data_iterator)):
            t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)

            if args.bf16:
                t_audio_signal_e = t_audio_signal_e.to(torch.bfloat16)
                t_transcript_e = t_transcript_e.to(torch.bfloat16)

            t_audio_signal_e_con = t_audio_signal_e.contiguous()

            # start E2E performance eval
            t0 = time.perf_counter()

            if args.xpu:
                t_audio_signal_e = t_audio_signal_e_con.to("xpu")
                t_a_sig_length_e = t_a_sig_length_e.to("xpu")
                t_transcript_e = t_transcript_e.to("xpu")
                t_transcript_len_e = t_transcript_len_e.to("xpu")

            t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args)

            # sync for time measurement
            torch.xpu.synchronize()

            t1 = time.perf_counter()

            if args.warm_up == 0 or it >= args.warm_up:
                total_time += (t1 - t0)

            print("sample id ", it, "  | cost time is ", t1 - t0)

            values_dict = dict(
                predictions=[t_predictions_e],
                transcript=[t_transcript_e],
                transcript_length=[t_transcript_len_e],
                # output=[t_log_probs_e]
            )
            process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

            if args.steps is not None and it + 1 >= (args.steps + args.warm_up):
                break

        hypotheses = _global_var_dict['predictions']
        references = _global_var_dict['transcripts']
        nb = len(hypotheses)
        print("print %d sample results: " % (min(len(hypotheses), nb)))
        for i, item in enumerate(hypotheses):
            print("hyp: ", hypotheses[i])
            print("ref: ", references[i])
            if i > nb:
                break

        wer, _ = process_evaluation_epoch(_global_var_dict)
        if (not multi_gpu or (multi_gpu and torch.distributed.get_rank() == 0)):
            print("\n========================>>>>>>>>>>>>>>>>>>>>>")
            print("\nEvaluation WER: {0}\n".format(wer))
            print("Accuracy: {:.15f} ".format(1 - wer))
            print()
            if args.save_prediction is not None:
                print("path is ", args.save_prediction)
                with open(args.save_prediction, 'w') as fp:
                    fp.write('\n'.join(_global_var_dict['predictions']))
            if logits_save_to is not None:
                logits = []
                for batch in _global_var_dict["logits"]:
                    for i in range(batch.shape[0]):
                        logits.append(batch[i].cpu().numpy())
                with open(logits_save_to, 'wb') as f:
                    pickle.dump(logits, f, protocol=pickle.HIGHEST_PROTOCOL)


        if args.steps:
            if args.steps * args.batch_size > len(data_layer):
                total_samples = len(data_layer)
            else:
                total_samples = args.steps * args.batch_size
        else:
            total_samples = len(data_layer)

        print("total samples tested: ", total_samples)
        print("total time (encoder + decoder, excluded audio processing): ", total_time, "s")
        print("dataset size: ", len(data_layer))

        perf = total_samples / total_time
        print("rnnt_inf throughput: {:.3f} sample per second".format(perf))
        print("rnnt_inf latency: ", total_time/total_samples)

def eval_profiling(
        data_layer,
        audio_processor,
        encoderdecoder,
        greedy_decoder,
        labels,
        multi_gpu,
        args):
    """performs inference / evaluation
    Args:
        data_layer: data layer object that holds data loader
        audio_processor: data processing module
        encoderdecoder: acoustic model
        greedy_decoder: greedy decoder
        labels: list of labels as output vocabulary
        multi_gpu: true if using multiple gpus
        args: script input arguments
    """
    logits_save_to=args.logits_save_to
    encoderdecoder.eval()
    with torch.no_grad():
        _global_var_dict = {
            'predictions': [],
            'transcripts': [],
            'logits' : [],
        }


        if args.wav:
            features, p_length_e = audio_processor(audio_from_file(args.wav))
            #torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_log_probs_e = encoderdecoder(features)
            #torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_predictions_e = greedy_decoder(log_probs=t_log_probs_e)
            hypotheses = __ctc_decoder_predictions_tensor(t_predictions_e, labels=labels)
            print("INFERENCE TIME\t\t: {} ms".format((t1-t0)*1000.0))
            print("TRANSCRIPT\t\t:", hypotheses[0])
            return
        print("\nstart measure performance, measure steps = ", args.steps)
        total_time = 0

        for it, data in enumerate(tqdm(data_layer.data_iterator)):
            t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)

            if args.bf16:
                t_audio_signal_e = t_audio_signal_e.to(torch.bfloat16)
                t_transcript_e = t_transcript_e.to(torch.bfloat16)

            t_audio_signal_e_con = t_audio_signal_e.contiguous()

            # start to eval profiling perf
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.XPU],
                record_shapes=False,
            ) as prof:
                try:
                    import memory_check
                    memory_check.display_mem("xpu:0")
                except:
                    pass
                t0 = time.perf_counter()
                if args.xpu:
                    t_audio_signal_e = t_audio_signal_e_con.to("xpu")
                    t_a_sig_length_e = t_a_sig_length_e.to("xpu")
                    t_transcript_e = t_transcript_e.to("xpu")
                    t_transcript_len_e = t_transcript_len_e.to("xpu")

                t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args)

            profiling_path = os.path.abspath('../') + '/report/'
            torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), profiling_path + 'rnnt_inference_profiling.pt')
            prof.export_chrome_trace(profiling_path + 'rnnt_inference_profiling.json')

            # sync for time measurement
            torch.xpu.synchronize()

            t1 = time.perf_counter()
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            # print(prof.key_averages(group_by_input_shape=True).table())
            # Cannot sort by id when using kineto
            # print(prof.table(sort_by="id", row_limit=100000))

            if args.warm_up == 0 or it >= args.warm_up:
                total_time += (t1 - t0)

            print("sample id ", it, "  | cost time is ", t1 - t0)

            values_dict = dict(
                predictions=[t_predictions_e],
                transcript=[t_transcript_e],
                transcript_length=[t_transcript_len_e],
                # output=[t_log_probs_e]
            )
            process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

            if args.steps is not None and it + 1 >= (args.steps + args.warm_up):
                break

        hypotheses = _global_var_dict['predictions']
        references = _global_var_dict['transcripts']
        nb = len(hypotheses)
        print("print %d sample results: " % (min(len(hypotheses), nb)))
        for i, item in enumerate(hypotheses):
            print("hyp: ", hypotheses[i])
            print("ref: ", references[i])
            if i > nb:
                break

        wer, _ = process_evaluation_epoch(_global_var_dict)
        if (not multi_gpu or (multi_gpu and torch.distributed.get_rank() == 0)):
            print("\n========================>>>>>>>>>>>>>>>>>>>>>")
            print("\nEvaluation WER: {0}\n".format(wer))
            print("Accuracy: {:.15f} ".format(1 - wer))
            print()
            if args.save_prediction is not None:
                print("path is ", args.save_prediction)
                with open(args.save_prediction, 'w') as fp:
                    
                    fp.write('\n'.join(_global_var_dict['predictions']))
            if logits_save_to is not None:
                logits = []
                for batch in _global_var_dict["logits"]:
                    for i in range(batch.shape[0]):
                        logits.append(batch[i].cpu().numpy())
                with open(logits_save_to, 'wb') as f:
                    pickle.dump(logits, f, protocol=pickle.HIGHEST_PROTOCOL)


        if args.steps:
            if args.steps * args.batch_size > len(data_layer):
                total_samples = len(data_layer)
            else:
                total_samples = args.steps * args.batch_size
        else:
            total_samples = len(data_layer)

        print("total samples tested: ", total_samples)
        print("total time (encoder + decoder, excluded audio processing): ", total_time, "s")
        print("dataset size: ", len(data_layer))

        perf = total_samples / total_time
        print("rnnt_inf throughput: {:.3f} sample per second".format(perf))
        print("rnnt_inf latency: ", total_time/total_samples)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.backends.cudnn.benchmark = args.cudnn_benchmark
    #print("CUDNN BENCHMARK ", args.cudnn_benchmark)
    #assert(torch.cuda.is_available())

    #if args.local_rank is not None:
    #    torch.cuda.set_device(args.local_rank)
    #    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    multi_gpu = args.local_rank is not None
    if multi_gpu:
        print("DISTRIBUTED with ", torch.distributed.get_world_size())

    if args.fp16:
        optim_level = Optimization.mxprO3
    else:
        optim_level = Optimization.mxprO0

    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    ctc_vocab = add_blank_label(dataset_vocab)

    val_manifest = args.val_manifest
    featurizer_config = model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level

    if args.max_duration is not None:
        featurizer_config['max_duration'] = args.max_duration
    if args.pad_to is not None:
        featurizer_config['pad_to'] = args.pad_to if args.pad_to >= 0 else "max"

    print('model_config')
    print_dict(model_definition)
    print('feature_config')
    print_dict(featurizer_config)
    data_layer = None
    
    if args.wav is None:
        data_layer = AudioToTextDataLayer(
            dataset_dir=args.dataset_dir, 
            featurizer_config=featurizer_config,
            manifest_filepath=val_manifest,
            labels=dataset_vocab,
            batch_size=args.batch_size,
            pad_to_max=featurizer_config['pad_to'] == "max",
            shuffle=False,
            multi_gpu=multi_gpu)
    audio_preprocessor = AudioPreprocessing(**featurizer_config)

    #encoderdecoder = JasperEncoderDecoder(jasper_model_definition=jasper_model_definition, feat_in=1024, num_classes=len(ctc_vocab))
    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(ctc_vocab)
    )
    print(model)

    if args.ckpt is not None:
        print("loading model from ", args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    #greedy_decoder = GreedyCTCDecoder()

    # print("Number of parameters in encoder: {0}".format(model.jasper_encoder.num_weights()))
    if args.wav is None:
        N = len(data_layer)
        step_per_epoch = math.ceil(N / (args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))

        if args.steps is not None:
            print('-----------------')
            print('Have {0} examples to eval on.'.format(args.steps * args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))
            print('Have {0} steps / (gpu * epoch).'.format(args.steps))
            print('-----------------')
        else:
            print('-----------------')
            print('Have {0} examples to eval on.'.format(N))
            print('Have {0} steps / (gpu * epoch).'.format(step_per_epoch))
            print('-----------------')
    else:
            audio_preprocessor.featurizer.normalize = "per_feature"

    print ("audio_preprocessor.normalize: ", audio_preprocessor.featurizer.normalize)

    audio_preprocessor.eval()

    eval_transforms = torchvision.transforms.Compose([
        lambda xs: [x.cpu() for x in xs],
        lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]],
        lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]],
    ])

    if args.xpu:
        model = model.to("xpu")
    else:
        model.cpu()
    if args.bf16:
        model = model.to(torch.bfloat16)

    # if args.channels_last:
    #     model = model.to(memory_format=torch.channels_last)


    if args.fp16:
        model = amp.initialize(
            models=model,
            opt_level=AmpOptimizations[optim_level])

    #model = model_multi_gpu(model, multi_gpu)

    greedy_decoder = RNNTGreedyDecoder(len(ctc_vocab) - 1, model.module if multi_gpu else model)

    if os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES","TRUE"]:
        eval_profiling(
            data_layer=data_layer,
            audio_processor=eval_transforms,
            encoderdecoder=model,
            greedy_decoder=greedy_decoder,
            labels=ctc_vocab,
            args=args,
            multi_gpu=multi_gpu)
    else:
        eval(
            data_layer=data_layer,
            audio_processor=eval_transforms,
            encoderdecoder=model,
            greedy_decoder=greedy_decoder,
            labels=ctc_vocab,
            args=args,
            multi_gpu=multi_gpu)
        

if __name__=="__main__":
    args = parse_args()
    print("xpu:", args.xpu)
    print("int8:", args.int8)
    print("bf16:", args.bf16)

    print_dict(vars(args))

    main(args)
