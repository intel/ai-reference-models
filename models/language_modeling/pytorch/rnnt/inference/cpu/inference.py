# Copyright (c) 2018 Intel Corporation
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

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

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
import random
import numpy as np
import pickle
import time

import torchvision


def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help='data batch size')
    parser.add_argument("--steps", default=None, help='if not specified do evaluation on full dataset. otherwise only evaluates the specified number of iterations for each worker', type=int)
    parser.add_argument("--model_toml", type=str, help='relative model configuration path given dataset folder')
    parser.add_argument("--dataset_dir", type=str, help='absolute path to dataset folder')
    parser.add_argument("--val_manifest", type=str, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--ckpt", default=None, type=str, required=True, help='path to model checkpoint')
    parser.add_argument("--max_duration", default=None, type=float, help='maximum duration of sequences. if None uses attribute from model configuration file')
    parser.add_argument("--pad_to", default=None, type=int, help="default is pad to value as specified in model configurations. if -1 pad to maximum duration. If > 0 pad batch to next multiple of value")
    parser.add_argument("--fp16", action='store_true', help='use half precision')
    parser.add_argument("--cudnn_benchmark", action='store_true', help="enable cudnn benchmark")
    parser.add_argument("--save_prediction", type=str, default=None, help="if specified saves predictions in text form at this location")
    parser.add_argument("--logits_save_to", default=None, type=str, help="if specified will save logits to path")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--wav", type=str, help='absolute path to .wav file (16KHz)')
    parser.add_argument("--warm_up", help='warm up steps, will only measure the performance from step=warm_up to step=(steps-warm_up)', type=int, default=0)
    parser.add_argument("--print-result", action='store_true', help='print prediction results', default=False)
    parser.add_argument("--print_time", action='store_true', help='print encoder decoder time', default=False)
    parser.add_argument("--ipex", action='store_true', help='use ipex', default=False)
    parser.add_argument("--int8", action='store_true', help='use int8', default=False)
    parser.add_argument("--jit", action='store_true', help='use jit', default=False)
    parser.add_argument("--mix-precision", action='store_true', help='use bf16', default=False)
    parser.add_argument("--profiling", action='store_true', help='do profiling', default=False)
    parser.add_argument("--sort_by_duration", action='store_true', help='sort sequence by duration', default=False)
    parser.add_argument('--calibration', action='store_true', default=False,
                    help='doing int8 calibration step')
    parser.add_argument('--configure-dir', default='configure.json', type=str, metavar='PATH',
                    help = 'path to int8 configures, default file name is configure.json')
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
    if args.ipex:
        import intel_extension_for_pytorch as ipex

    logits_save_to=args.logits_save_to
    encoderdecoder.eval()
    with torch.no_grad():
        _global_var_dict = {
            'predictions': [],
            'transcripts': [],
            'logits' : [],
        }

        if args.wav:
            # TODO unimplemented in ipex
            assert False, "wav unsupported in ipex for now"
            features, p_length_e = audio_processor(audio_from_file(args.wav))
            # torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_log_probs_e = encoderdecoder(features)
            # torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_predictions_e = greedy_decoder(log_probs=t_log_probs_e)
            hypotheses = __ctc_decoder_predictions_tensor(t_predictions_e, labels=labels)
            print("INFERENCE TIME\t\t: {} ms".format((t1-t0)*1000.0))
            print("TRANSCRIPT\t\t:", hypotheses[0])
            return
        steps_per_epoch = len(data_layer)
        total_steps = args.steps if args.steps is not None else steps_per_epoch
        test_epoches = int(total_steps / steps_per_epoch)
        print('Evaluating RNNT: Steps per Epoch {} total Steps {}'.format(steps_per_epoch, total_steps))

        # Int8 Calibration
        if args.ipex and args.int8 and args.calibration:
            print("runing int8 calibration step\n")
            conf = ipex.AmpConf(torch.int8)            
            for it, data in enumerate(tqdm(data_layer.data_iterator)):
                t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)
                
                t_predictions_e, conf = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)

                if args.steps is not None and it + 1 >= args.steps:
                    break
            conf.save(args.configure_dir)
        # Inference (vanilla cpu, dnnl fp32 or dnnl int8)
        else:
            if not args.ipex:
                if args.warm_up > 0:
                    print("\nstart warm up, warmp_up steps = ", args.warm_up)
                    for it, data in enumerate(tqdm(data_layer.data_iterator)):
                            t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)
                            conf = None
                            t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                            
                            if it + 1 >= args.warm_up:
                                break
                print("\nstart measure performance, measure steps = ", total_steps)
                total_time = 0
                with tqdm(total=total_steps) as pbar:
                    for epoch in range(test_epoches + 1):
                        for it, data in enumerate(data_layer.data_iterator):
                            if  epoch * steps_per_epoch + it >= total_steps:
                                break
                            t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)
                            if args.profiling:
                                # with torch.autograd.profiler.profile(args.profiling) as prof:
                                with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
                                    conf = None
                                    t0 = time.perf_counter()
                                    t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                                    t1 = time.perf_counter()
                            else:
                                conf = None
                                t0 = time.perf_counter()
                                t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                                t1 = time.perf_counter()

                            total_time += (t1 - t0)

                            values_dict = dict(
                                predictions=[t_predictions_e],
                                transcript=[t_transcript_e],
                                transcript_length=[t_transcript_len_e],
                            )
                            process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

                            pbar.update(1)
            else:
                if args.mix_precision:
                    with torch.cpu.amp.autocast():
                        # warm up
                        if args.warm_up > 0:
                            print("\nstart warm up, warmp_up steps = ", args.warm_up)
                            for it, data in enumerate(tqdm(data_layer.data_iterator)):
                                t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)
                                conf = None
                                t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                                
                                if it + 1 >= args.warm_up:
                                    break

                        # measure performance
                        print("\nstart measure performance, measure steps = ", total_steps)
                        total_time = 0
                        # with torch.autograd.profiler.profile(args.profiling) as prof:
                        with tqdm(total=total_steps) as pbar:
                            for epoch in range(test_epoches + 1):
                                for it, data in enumerate(data_layer.data_iterator):
                                    if epoch * steps_per_epoch + it >= total_steps:
                                        break
                                    t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)
                                    if args.profiling:
                                        # with torch.autograd.profiler.profile(args.profiling) as prof:
                                        with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
                                            conf = None
                                            t0 = time.perf_counter()
                                            t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                                            t1 = time.perf_counter()
                                    else:
                                        conf = None
                                        t0 = time.perf_counter()
                                        t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                                        t1 = time.perf_counter()

                                    total_time += (t1 - t0)

                                    values_dict = dict(
                                        predictions=[t_predictions_e],
                                        transcript=[t_transcript_e],
                                        transcript_length=[t_transcript_len_e],
                                    )
                                    process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

                                    pbar.update(1)
                else:
                    # warm up
                    if args.warm_up > 0:
                        print("\nstart warm up, warmp_up steps = ", args.warm_up)
                        for it, data in enumerate(tqdm(data_layer.data_iterator)):
                            t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)
                            conf = None
                            t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                            
                            if it + 1 >= args.warm_up:
                                break

                    # measure performance
                    print("\nstart measure performance, measure steps = ", total_steps)
                    total_time = 0
                    # with torch.autograd.profiler.profile(args.profiling) as prof:
                    with tqdm(total=total_steps) as pbar:
                        for epoch in range(test_epoches + 1):
                            for it, data in enumerate(data_layer.data_iterator):
                                if epoch * steps_per_epoch + it >= total_steps:
                                    break
                                t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = audio_processor(data)
                                if args.profiling:
                                    # with torch.autograd.profiler.profile(args.profiling) as prof:
                                    with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
                                        conf = None
                                        t0 = time.perf_counter()
                                        t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                                        t1 = time.perf_counter()
                                else:
                                    conf = None
                                    t0 = time.perf_counter()
                                    t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e, args, conf)
                                    t1 = time.perf_counter()

                                total_time += (t1 - t0)

                                values_dict = dict(
                                    predictions=[t_predictions_e],
                                    transcript=[t_transcript_e],
                                    transcript_length=[t_transcript_len_e],
                                )
                                process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

                                pbar.update(1)

            if args.print_result:
                hypotheses = _global_var_dict['predictions']
                references = _global_var_dict['transcripts']

                nb = len(hypotheses)
                print("print %d sample results: " % (min(len(hypotheses), nb)))
                for i, item in enumerate(hypotheses):
                    print("hyp: ", hypotheses[i])
                    print("ref: ", references[i])
                    print()
                    if i > nb:
                        break
            
            if args.profiling:
                # print(prof.key_averages().table(sort_by="cpu_time_total"))
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))

            wer, _ = process_evaluation_epoch(_global_var_dict)
            if (not multi_gpu or (multi_gpu and torch.distributed.get_rank() == 0)):
                print("\n=========================>>>>>>")
                print("Evaluation WER: {0}".format(wer))
                print("Accuracy: {:.15f} ".format(1 - wer))
                if args.save_prediction is not None:
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
                total_samples = args.steps * args.batch_size
            else:
                total_samples = len(data_layer)

            print("total samples tested: ", total_samples)
            print("total time (encoder + decoder, excluded audio processing): ", total_time, "s")
            print("dataset size: ", len(data_layer))

            perf = total_samples / total_time

            print("Throughput: {:.3f} fps".format(perf))

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

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
            # sampler='bucket',
            sort_by_duration=args.sort_by_duration,
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

    if args.ckpt is not None:
        print("loading model from ", args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.ipex:
        import intel_extension_for_pytorch as ipex
        from rnn import IPEXStackTime
        model.joint_net.eval()
        data_type = torch.bfloat16 if args.mix_precision else torch.float32
        if model.encoder["stack_time"].factor == 2:
            model.encoder["stack_time"] = IPEXStackTime(model.encoder["stack_time"].factor)
        model.joint_net = ipex.optimize(model.joint_net, dtype=data_type, auto_kernel_selection=True)
        model.prediction["embed"] = model.prediction["embed"].to(data_type)
        if args.jit:
            print("running jit path")
            model.joint_net.eval()
            if args.mix_precision:
                with torch.cpu.amp.autocast(), torch.no_grad():
                    model.joint_net = torch.jit.trace(model.joint_net, torch.randn(args.batch_size, 1, 1, model_definition['rnnt']['encoder_n_hidden'] + model_definition['rnnt']['pred_n_hidden']), check_trace=False)
            else:
                with torch.no_grad():
                    model.joint_net = torch.jit.trace(model.joint_net, torch.randn(args.batch_size, 1, 1, model_definition['rnnt']['encoder_n_hidden'] + model_definition['rnnt']['pred_n_hidden']), check_trace=False)
            model.joint_net = torch.jit.freeze(model.joint_net)
    else:
        model = model.to("cpu")

    #greedy_decoder = GreedyCTCDecoder()

    # print("Number of parameters in encoder: {0}".format(model.jasper_encoder.num_weights()))
    if args.wav is None:
        N = len(data_layer)
        # step_per_epoch = math.ceil(N / (args.batch_size * (1 if not torch.distributed.is_available() else torch.distributed.get_world_size())))
        step_per_epoch = math.ceil(N / (args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))

        if args.steps is not None:
            print('-----------------')
            # print('Have {0} examples to eval on.'.format(args.steps * args.batch_size * (1 if not torch.distributed.is_available() else torch.distributed.get_world_size())))
            print('Have {0} examples to eval on.'.format(args.steps * args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))
            print('Have {0} warm up steps / (gpu * epoch).'.format(args.warm_up))
            print('Have {0} measure steps / (gpu * epoch).'.format(args.steps))
            print('-----------------')
        else:
            print('-----------------')
            print('Have {0} examples to eval on.'.format(N))
            print('Have {0} warm up steps / (gpu * epoch).'.format(args.warm_up))
            print('Have {0} measure steps / (gpu * epoch).'.format(step_per_epoch))
            print('-----------------')
    else:
            audio_preprocessor.featurizer.normalize = "per_feature"

    print ("audio_preprocessor.normalize: ", audio_preprocessor.featurizer.normalize)
    audio_preprocessor.eval()

    # eval_transforms = torchvision.transforms.Compose([
    #     lambda xs: [x.to(ipex.DEVICE) if args.ipex else x.cpu() for x in xs],
    #     lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]],
    #     lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]],
    # ])

    eval_transforms = torchvision.transforms.Compose([
        lambda xs: [x.cpu() for x in xs],
        lambda xs: [*audio_preprocessor(xs[0:2]), *xs[2:]],
        lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]],
    ])

    model.eval()
    if args.ipex:
        ipex.nn.utils._model_convert.replace_lstm_with_ipex_lstm(model)

    greedy_decoder = RNNTGreedyDecoder(len(ctc_vocab) - 1, model.module if multi_gpu else model)

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

    print_dict(vars(args))

    main(args)
