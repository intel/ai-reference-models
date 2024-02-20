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

import argparse
import math
import os
import random
import time
from heapq import nlargest
from itertools import chain, repeat
from pathlib import Path
from tqdm import tqdm

import torch
import intel_extension_for_pytorch as ipex
import numpy as np

from quartznet import config
from common import helpers
from common.dataset import (AudioDataset, FilelistDataset, get_data_loader,
                            SingleAudioDataset)
from common.features import BaseFeatures, FilterbankFeatures
from common.helpers import print_once, process_evaluation_epoch
from quartznet.model import GreedyCTCDecoder, QuartzNet


def get_parser():
    parser = argparse.ArgumentParser(description='QuartzNet inference')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Data batch size')
    parser.add_argument('--dynamic_shape', default=False, action='store_true', help='use dynamic shape')
    parser.add_argument('--steps', default=10, type=int,
                        help='Eval this many steps for every worker')
    parser.add_argument('--warmup_steps', default=5, type=int,
                        help='Burn-in period before measuring latencies')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Relative model config path given dataset folder')
    parser.add_argument('--dataset_dir', type=str,
                        help='Absolute path to dataset folder')
    parser.add_argument('--val_manifests', type=str, nargs='+',
                        help='Relative path to evaluation dataset manifest files')
    parser.add_argument('--ckpt', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--amp', '--fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--cudnn_benchmark', default = False, action='store_true',
                        help='Enable cudnn benchmark')
    parser.add_argument('--cpu', default = False, action='store_true',
                        help='Run inference on CPU')
    parser.add_argument('--xpu', default = True, action='store_true',
                        help='Run inference on XPU')
    parser.add_argument('--channels_last', default = True, action='store_true',
                        help='Use channels last')
    parser.add_argument("--seed", default=0, type=int, help='Random seed')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0),
                        type=int, help='GPU id used for distributed training')                     

    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--dali_device', type=str, choices=['none', 'cpu', 'gpu'],
                    default='none', help='Use DALI pipeline for fast data processing')
    io.add_argument('--save_predictions', type=str, default=None,
                    help='Save predictions in text form at this location')
    io.add_argument('--save_logits', default=None, type=str,
                    help='Save output logits under specified path')
    io.add_argument('--transcribe_wav', type=str,
                    help='Path to a single .wav file (16KHz)')
    io.add_argument('--transcribe_filelist', type=str,
                    help='Path to a filelist with one .wav path per line')
    io.add_argument('-o', '--output_dir', default='results/',
                    help='Output folder to save audio (file per phrase)')
    io.add_argument('--log_file', type=str, default=None,
                    help='Path to a DLLogger log file')
    io.add_argument('--ema', action='store_true', default=True,
                    help='Load averaged model weights')
    io.add_argument('--torchscript', action='store_true', default=True,
                    help='Evaluate with a TorchScripted model')
    io.add_argument('--torchscript_export', action='store_true', default=True,
                    help='Export the model with torch.jit to the output_dir')
    io.add_argument('--override_config', type=str, action='append',
                    help='Overrides arbitrary config value.'
                         ' Syntax: `--override_config nested.config.key=val`.')
    io.add_argument('--benchmark', action='store_true',
                        help='Enable benchmark')
    return parser

def torchscript_export(data_loader, audio_processor, model, greedy_decoder,
                       output_dir, use_amp, use_conv_masks, model_config, device,
                       save):

    audio_processor = audio_processor.to(device)

    for batch in data_loader:
        batch = [t.to(device) for t in batch]
        audio, audio_len, _, _ = batch
        feats, feat_lens = audio_processor(audio, audio_len)
        break

    print("\nExporting featurizer...")
    print("\nNOTE: Dithering causes warnings about non-determinism.\n")
    ts_feat = torch.jit.trace(audio_processor, (audio, audio_len), check_trace=False)

    print("\nExporting acoustic model...")
    if device.type == 'xpu' and use_amp:
        with torch.no_grad():
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
                ts_acoustic = torch.jit.trace(model, (feats, feat_lens), check_trace=False)
    else:
        ts_acoustic = torch.jit.trace(model, (feats, feat_lens))

    print("\nExporting decoder...")
    log_probs = model(feats, feat_lens)
    ts_decoder = torch.jit.script(greedy_decoder, log_probs)
    print("\nJIT export complete.")

    if save:
        precision = "fp16" if use_amp else "fp32"
        module_name = f'{os.path.basename(model_config)}_{precision}'
        ts_feat.save(os.path.join(output_dir, module_name + "_feat.pt"))
        ts_acoustic.save(os.path.join(output_dir, module_name + "_acoustic.pt"))
        ts_decoder.save(os.path.join(output_dir, module_name + "_decoder.pt"))

    return ts_feat, ts_acoustic, ts_decoder


def main():

    parser = get_parser()
    args = parser.parse_args()

    args.profiling= os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

    if not args.cpu and not args.xpu:
        from common.dali.data_loader import DaliDataLoader
        import dllogger
        from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
        from common.tb_dllogger import stdout_metric_format, unique_log_fpath

        log_fpath = args.log_file or str(Path(args.output_dir, 'nvlog_infer.json'))
        dllogger.init(backends=[
            JSONStreamBackend(Verbosity.DEFAULT, log_fpath, append=True),
            JSONStreamBackend(Verbosity.DEFAULT, unique_log_fpath(log_fpath)),
            StdOutBackend(Verbosity.VERBOSE, metric_format=stdout_metric_format)
        ])
        [dllogger.log("PARAMETER", {k: v}) for k, v in vars(args).items()]

        for step in ['DNN', 'data+DNN', 'data']:
            for c in [0.99, 0.95, 0.9, 0.5]:
                cs = 'avg' if c == 0.5 else f'{int(100*c)}%'
                dllogger.metadata(f'{step.lower()}_latency_{c}',
                                {'name': f'{step} latency {cs}',
                                'format': ':>7.2f', 'unit': 'ms'})
        dllogger.metadata(
            'eval_wer', {'name': 'WER', 'format': ':>3.2f', 'unit': '%'})

    if args.cpu:
        device = torch.device('cpu')
    elif args.xpu:
        device = torch.device('xpu')
    else:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.seed is not None:
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)
        if args.xpu:
            torch.xpu.manual_seed(args.seed + args.local_rank)

    # set up distributed training
    multi_gpu = not args.cpu and int(os.environ.get('WORLD_SIZE', 1)) > 1
    
    # multi gpu in only for cuda now
    if multi_gpu:
        import torch.distributed as distrib
        torch.cuda.set_device(args.local_rank)
        distrib.init_process_group(backend='nccl', init_method='env://')
        print_once(f'Inference with {distrib.get_world_size()} GPUs')

    # load weight of pre-trained model
    if args.ckpt is not None:
        print(f'Loading the model from {args.ckpt} ...')
        print(f'{args.model_config} will be overriden.')
        if args.ckpt.lower().endswith('.nemo'):
            from nemo_dle_model_converter import load_nemo_ckpt
            ckpt, cfg = load_nemo_ckpt(args.ckpt)
        else:
            cfg = config.load(args.model_config)
            ckpt = torch.load(args.ckpt, map_location='cpu')

        sd_key = 'ema_state_dict' if args.ema else 'state_dict'
        if args.ema and 'ema_state_dict' not in ckpt:
            print(f'WARNING: EMA weights are unavailable in {args.ckpt}.')
            sd_key = 'state_dict'
        state_dict = ckpt[sd_key]

    else:
        cfg = config.load(args.model_config)
        state_dict = None

    config.apply_config_overrides(cfg, args)

    symbols = helpers.add_ctc_blank(cfg['labels'])

    use_dali = args.dali_device in ('cpu', 'gpu')
    dataset_kw, features_kw = config.input(cfg, 'val')

    measure_perf = args.steps > 0

    # prepare dataset and initialize feature processor
    dataset_kw['dynamic_shape'] = args.dynamic_shape
    if args.transcribe_wav or args.transcribe_filelist:

        if use_dali:
            print("DALI supported only with input .json files; disabling")
            use_dali = False

        assert not cfg['input_val']['audio_dataset'].get('pad_to_max_duration', False)
        assert not (args.transcribe_wav and args.transcribe_filelist)

        if args.transcribe_wav:
            dataset = SingleAudioDataset(args.transcribe_wav)
        else:
            dataset = FilelistDataset(args.transcribe_filelist)

        data_loader = get_data_loader(dataset,
                                      batch_size=1,
                                      multi_gpu=multi_gpu,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=(True if measure_perf else False))

        _, features_kw = config.input(cfg, 'val')
        feat_proc = FilterbankFeatures(**features_kw)

    elif use_dali:
        # pad_to_max_duration is not supported by DALI - have simple padders
        if features_kw['pad_to_max_duration']:
            feat_proc = BaseFeatures(
                pad_align=features_kw['pad_align'],
                pad_to_max_duration=True,
                max_duration=features_kw['max_duration'],
                sample_rate=features_kw['sample_rate'],
                window_size=features_kw['window_size'],
                window_stride=features_kw['window_stride'])
            features_kw['pad_to_max_duration'] = False
        else:
            feat_proc = None

        data_loader = DaliDataLoader(
            gpu_id=args.local_rank or 0,
            dataset_path=args.dataset_dir,
            config_data=dataset_kw,
            config_features=features_kw,
            json_names=args.val_manifests,
            batch_size=args.batch_size,
            pipeline_type=("train" if measure_perf else "val"),  # no drop_last
            device_type=args.dali_device,
            symbols=symbols)

    else:
        print(dataset_kw)
        dataset = AudioDataset(args.dataset_dir,
                               args.val_manifests,
                               symbols,
                               **dataset_kw)

        data_loader = get_data_loader(dataset,
                                      args.batch_size,
                                      multi_gpu=multi_gpu,
                                      shuffle=False,
                                      num_workers=4,
                                      drop_last=True)

        feat_proc = FilterbankFeatures(**features_kw)

    # initilaize model
    model = QuartzNet(encoder_kw=config.encoder(cfg),
                      decoder_kw=config.decoder(cfg, n_classes=len(symbols)))

    # load model config
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)

    # prepare model and feat_proc for jit trace
    model = model.to(device)
    model.eval()

    if feat_proc is not None:
        feat_proc = feat_proc.to(device)
        feat_proc.eval()

    if args.amp and device.type == 'cuda':
        model = model.half()
    
    if device.type == 'xpu':
        if args.channels_last:
            ipex.enable_auto_channels_last()
        if args.amp:
            model = torch.xpu.optimize(model, dtype=torch.half)
        else:
            model = torch.xpu.optimize(model, dtype=torch.float)

    # initialize greedy_decoder
    greedy_decoder = GreedyCTCDecoder()

    # record if use_conv_masks in model, since this information gets lost after jit trace
    mask = model.encoder.use_conv_masks
    
    # jit trace for feat_proc, model
    # jit script for greedy_decoder
    if args.torchscript:
        feat_proc, model, greedy_decoder = torchscript_export(
            data_loader, feat_proc, model, greedy_decoder, args.output_dir,
            use_amp=args.amp, use_conv_masks=True, model_config=args.model_config,
            device=device, save=args.torchscript_export)

    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    agg = {'txts': [], 'preds': [], 'logits': []}

    #looped_loader = chain.from_iterable(repeat(data_loader))

    if device.type == 'cuda':
        sync = lambda: torch.cuda.synchronize()
    elif device.type == 'xpu':
        sync = lambda: torch.xpu.synchronize()
    else:
        sync = None

    total_time = 0
    steps = args.steps + args.warmup_steps # or len(data_loader)
    with torch.no_grad():

        #for it, batch in enumerate(tqdm(looped_loader, initial=1, total=steps)):
        for it, batch in enumerate(data_loader):
            with torch.autograd.profiler_legacy.profile(enabled=args.profiling, use_xpu=True, record_shapes=False) as prof:
                print("Iteration: ", it)
                start_time = time.time()

                # H2D + encoder
                if use_dali:
                    feats, feat_lens, txt, txt_lens = batch
                    if feat_proc is not None:
                        feats, feat_lens = feat_proc(feats, feat_lens)
                else:
                    batch = [t.to(device) for t in batch]
                    audio, audio_lens, txt, txt_lens = batch
                    feats, feat_lens = feat_proc(audio, audio_lens)

                if mask:
                    # fp16 with imperative path
                    if args.amp and not args.torchscript:
                        with torch.xpu.amp.autocast(enabled=True, dtype=torch.torch.float16, cache_enabled=False):
                            log_probs, log_prob_lens = model(feats, feat_lens)
                    else:
                        log_probs, log_prob_lens = model(feats, feat_lens)
                else:
                    # fp16 with imperative path
                    if args.amp and not args.torchscript:
                        with torch.xpu.amp.autocast(enabled=True, dtype=torch.torch.float16, cache_enabled=False):
                            log_probs = model(feats, feat_lens)
                    else:
                        log_probs = model(feats, feat_lens)

                # Decoder
                preds = greedy_decoder(log_probs)
                preds = preds.to("cpu")

                sync()
                end_time = time.time()
                if it >= args.warmup_steps:
                    total_time += end_time - start_time
                    print("Time used: " + str(end_time - start_time) + ". Input shape: " + str(feats.shape))
                    latency = total_time / (it - args.warmup_steps + 1)
                    throughput = args.batch_size / latency
                    if it + 1 == steps:
                        print("---latency={} s".format(latency))
                        print("---throughput={} fps".format(throughput))

            if args.profiling:  
                title = "/quartznet_inference_"  
                if args.amp:
                    title += "fp16_"
                if args.batch_size:
                    title += "bs" + str(args.batch_size) + "_"

                profiling_path = args.output_dir
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), profiling_path + title + 'profiling.pt')
                prof.export_chrome_trace(profiling_path + title + 'profiling.json')
                print(prof.key_averages().table(sort_by="self_xpu_time_total"))
                print(prof.key_averages(group_by_input_shape=True).table())
                # print(prof.table(sort_by="id", row_limit=100000))

            if txt is not None:
                agg['txts'] += helpers.gather_transcripts([txt], [txt_lens],
                                                          symbols)
            agg['preds'] += helpers.gather_predictions([preds], symbols)
            agg['logits'].append(log_probs)

            if it + 1 == steps:
                break

        # communicate the results
        if args.transcribe_wav:
            for idx, p in enumerate(agg['preds']):
                print_once(f'Prediction {idx+1: >3}: {p}')

        elif args.transcribe_filelist:
            pass

        elif (not multi_gpu or distrib.get_rank() == 0) and args.benchmark == 0:
            wer, _ = process_evaluation_epoch(agg)

            # dllogger.log(step=(), data={'eval_wer': 100 * wer})
            print("WER={} %".format(100 * wer))

        if args.save_predictions:
            with open(args.save_predictions, 'w') as f:
                f.write('\n'.join(agg['preds']))

        if args.save_logits:
            logits = torch.cat(agg['logits'], dim=0).cpu()
            torch.save(logits, args.save_logits)



if __name__ == "__main__":
    main()
