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
import contextlib
import itertools
import os
import time
import toml
import torch
import sys
#import apex
#from apex import amp
import random
import numpy as np
import math
from dataset import AudioToTextDataLayer
from helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch, Optimization, add_blank_label, AmpOptimizations, print_dict, print_once
from model_rnnt import RNNT
from decoders import RNNTGreedyDecoder
from loss import RNNTLoss
from loss import TransducerLoss
from optimizers import Novograd, AdamW

import torchvision

from tb_logger import DummyLogger, TensorBoardLogger
import preprocessing

import os
import intel_extension_for_pytorch
from tqdm import tqdm


def lr_decay(N, step, learning_rate):
    """
    learning rate decay
    Args:
        learning_rate: base learning rate
        step: current iteration number
        N: total number of iterations over which learning rate is decayed
    """
    min_lr = 0.00001
    res = learning_rate * ((N - step) / N) ** 2
    return max(res, min_lr)

def lr_warmup(warmup_steps, step, learning_rate):
    return min(1, (step / warmup_steps)) * learning_rate

def save(model, optimizer, epoch, output_dir):
    """
    Saves model checkpoint
    Args:
        model: model
        optimizer: optimizer
        epoch: epoch of model training
        output_dir: path to save model checkpoint
    """
    class_name = model.__class__.__name__
    unix_time = time.time()
    file_name = "{0}_{1}-epoch-{2}.pt".format(class_name, unix_time, epoch)
    print_once("Saving module {0} in {1}".format(class_name, os.path.join(output_dir, file_name)))
    if (not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        save_checkpoint={
                        'epoch': epoch,
                        'state_dict': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }

        torch.save(save_checkpoint, os.path.join(output_dir, file_name))
    print_once('Saved.')


def evaluator(model, data_transforms, loss_fn, greedy_decoder, labels, eval_datasets, logger):
    """Evaluates model on evaluation dataset
    """

    def evalutaion(epoch=0):
        model.eval()

        for dataset, frequency, name in eval_datasets:
            if epoch % frequency != 0:
                continue

            print_once(f"Doing {name} ....................... ......  ... .. . .")

            with torch.no_grad():
                _global_var_dict = {
                    'EvalLoss': [],
                    'predictions': [],
                    'transcripts': [],
                }
                dataloader = dataset.data_iterator
                for data in dataloader:
                    t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = data_transforms(data)

                    if args.xpu:
                        t_audio_signal_e_con = t_audio_signal_e.contiguous()
                        t_audio_signal_e = t_audio_signal_e_con.to(args.xpu_id)
                        t_a_sig_length_e = t_a_sig_length_e.to(args.xpu_id)
                        t_transcript_e = t_transcript_e.to(args.xpu_id)
                        t_transcript_len_e = t_transcript_len_e.to(args.xpu_id)

                    if args.bf16:
                        t_audio_signal_e = t_audio_signal_e.to(torch.bfloat16)
                        t_transcript_e = t_transcript_e.to(torch.bfloat16)


                    t_log_probs_e, (x_len, y_len) = model(
                        ((t_audio_signal_e, t_transcript_e), (t_a_sig_length_e, t_transcript_len_e)),
                    )
                    t_loss_e = loss_fn(
                        (t_log_probs_e, x_len), (t_transcript_e, y_len)
                    )
                    del t_log_probs_e

                    t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e)



                    values_dict = dict(
                        loss=[t_loss_e],
                        predictions=[t_predictions_e],
                        transcript=[t_transcript_e],
                        transcript_length=[t_transcript_len_e]
                    )
                    process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

                # final aggregation across all workers and minibatches) and logging of results
                wer, eloss = process_evaluation_epoch(_global_var_dict)
                logger.log_scalar('loss', eloss, epoch, name)
                logger.log_scalar('wer', wer, epoch, name)

                print_once(f"==========>>>>>>{name} Loss: {eloss}\n")
                print_once(f"==========>>>>>>{name} WER: {wer}\n")

    return evalutaion


def train(
        data_layer,
        model,
        loss_fn,
        greedy_decoder,
        optimizer,
        optim_level,
        labels,
        multi_gpu,
        data_transforms,
        args,
        evalutaion,
        logger,
        profiling,
        fn_lr_policy):
    """Trains model
    Args:
        data_layer: training data layer
        model: model ( encapsulates data processing, encoder, decoder)
        loss_fn: loss function
        greedy_decoder: greedy ctc decoder
        optimizer: optimizer
        optim_level: AMP optimization level
        labels: list of output labels
        multi_gpu: true if multi gpu training
        args: script input argument list
        fn_lr_policy: function returning lr in given step
    """
    print_once("Starting .....")
    start_time = time.time()

    train_dataloader = data_layer.data_iterator
    epoch = args.start_epoch
    step = epoch * args.step_per_epoch

    while True:
        if multi_gpu:
            data_layer.sampler.set_epoch(epoch)
        print_once("Starting epoch {0}, step {1}".format(epoch, step))
        last_epoch_start = time.time()
        batch_counter = 0
        average_loss = 0
        total_time = 0
        loss_cpu = torch.tensor(0, device='cpu')

        for data in tqdm(train_dataloader):
            if args.cuda and optim_level in AmpOptimizations:
                assert False, "not supported in intel_extension_for_pytorch"
            if batch_counter == 0:

                adjusted_lr = fn_lr_policy(step)
                for param_group in optimizer.param_groups:
                        param_group['lr'] = adjusted_lr
                optimizer.zero_grad(set_to_none=True)
                last_iter_start = time.time()

            t_audio_signal_t, t_a_sig_length_t, t_transcript_t, t_transcript_len_t = data_transforms(data)

            model.train()

            if args.bf16:
                t_audio_signal_t = t_audio_signal_t.to(torch.bfloat16)
                t_transcript_t = t_transcript_t.to(torch.bfloat16)

            t_audio_signal_t_con = t_audio_signal_t.contiguous()

            with (
                contextlib.nullcontext(None) if not profiling else
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.XPU],
                    record_shapes=False,
                )
            ) as prof:
                try:
                    import memory_check
                    memory_check.display_mem("xpu:0")
                except:
                    pass
                t0 = time.perf_counter()
                if args.xpu:
                    t_audio_signal_t = t_audio_signal_t_con.to(args.xpu_id)
                    t_a_sig_length_t = t_a_sig_length_t.to(args.xpu_id)
                    t_transcript_t = t_transcript_t.to(args.xpu_id)
                    t_transcript_len_t = t_transcript_len_t.to(args.xpu_id)

                if args.batch_split_factor == 1:
                    t_log_probs_t, (x_len, y_len) = model(
                        ((t_audio_signal_t, t_transcript_t), (t_a_sig_length_t, t_transcript_len_t)),
                    )

                    t_loss_t = loss_fn(t_log_probs_t, x_len, t_transcript_t, y_len)
                    # t_loss_t = loss_fn(args,
                    #                    (t_log_probs_t, x_len), (t_transcript_t, y_len))
                    if torch.isnan(t_loss_t).any():
                        raise Exception("Loss is NaN")

                    if args.cuda and optim_level in AmpOptimizations:
                        assert False, "not supported in intel_extension_for_pytorch"

                    # if optim_level in AmpOptimizations:
                    #     with amp.scale_loss(t_loss_t, optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    else:
                        t_loss_t.backward()
                else:
                    if args.distributed:
                        f, g, t_log_probs_t = model.module.enc_pred(((t_audio_signal_t, t_transcript_t), (t_a_sig_length_t, t_transcript_len_t)),)
                    else:
                        f, g, t_log_probs_t = model.enc_pred(((t_audio_signal_t, t_transcript_t), (t_a_sig_length_t, t_transcript_len_t)),)
                    f_2, g_2 = f.detach(), g.detach()
                    f_2.requires_grad = True
                    g_2.requires_grad = True
                    B_split = args.batch_size // args.batch_split_factor
                    t_loss_t = 0
                    for i in range(args.batch_split_factor):
                        if args.distributed:
                            log_probs = model.module.joint(f_2[i*B_split:(i+1)*B_split], g_2[i*B_split:(i+1)*B_split], 
                                                    t_log_probs_t[i*B_split:(i+1)*B_split])
                        else:
                            log_probs = model.joint(f_2[i*B_split:(i+1)*B_split], g_2[i*B_split:(i+1)*B_split], 
                                                    t_log_probs_t[i*B_split:(i+1)*B_split])
                        loss = loss_fn(log_probs, t_log_probs_t[i*B_split:(i+1)*B_split],
                                        t_transcript_t[i*B_split:(i+1)*B_split], t_transcript_len_t[i*B_split:(i+1)*B_split] )
                        loss /= (args.gradient_accumulation_steps*args.batch_split_factor)
                        if torch.isnan(loss).any():
                            raise Exception("Loss is NaN")
                        loss_cpu.copy_(loss.detach())
                        del log_probs
                        loss.backward()
                        t_loss_t +=  loss_cpu.item()
                    f.backward(f_2.grad)
                    g.backward(g_2.grad)

                if batch_counter % args.gradient_accumulation_steps == 0:
                    optimizer.step()

                # sync for time measurement
                torch.xpu.synchronize()
            if profiling:
                profile_name = 'fp32'
                if args.fp16:
                    profile_name = 'fp16'
                elif args.bf16:
                    profile_name = 'bf16'
                if args.distributed:
                    profile_name += '.xpu.' + str(args.rank)
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './rnnt_profiling.' + profile_name + '.train.pt')
                # Cannot sort by id when using kineto
                # torch.save(prof.table(sort_by="id", row_limit=100000), './rnnt_profiling.' + profile_name + '.train.detailed.pt')
                prof.export_chrome_trace('./rnnt_profiling' + profile_name + '.json')

            t1 = time.perf_counter()

            if args.warm_up == 0 or step >= args.warm_up:
                total_time += (t1 - t0)
            print("sample id ", step, "  | cost time is ", t1 - t0)

            logger.log_scalar('loss', t_loss_t, step)

            del t_log_probs_t
            if args.gradient_accumulation_steps > 1:
                t_loss_t = t_loss_t / args.gradient_accumulation_steps

            batch_counter += 1
            average_loss += t_loss_t

            if batch_counter % args.gradient_accumulation_steps == 0:
                if (step + 1) % args.train_frequency == 0:
                    # t_predictions_t = greedy_decoder.decode(t_audio_signal_t, t_a_sig_length_t)

                    # e_tensors = [t_predictions_t, t_transcript_t, t_transcript_len_t]
                    # train_wer = monitor_asr_train_progress(e_tensors, labels=labels)
                    print_once("Loss@Step: {0}  ::::::: {1}".format(step, str(average_loss)))
                    print_once("Step time: {0} seconds".format(time.time() - last_iter_start))
                    # logger.log_scalar('wer', train_wer, step)

                step += 1
                batch_counter = 0
                average_loss = 0
                if args.num_steps is not None and step >= (args.num_steps + args.warm_up):
                    break

            print("====t_loss_t===", t_loss_t)
            torch.xpu.empty_cache()

        # evalutaion(epoch)

        if args.num_steps is not None and step >= args.num_steps:
            break
        print_once("Finished epoch {0} in {1}".format(epoch, time.time() - last_epoch_start))
        epoch += 1
        if epoch % args.save_frequency == 0 and epoch > 0:
            save(model, optimizer, epoch, output_dir=args.output_dir)
        if args.num_steps is None and epoch >= args.num_epochs:
            break

    if args.num_steps is not None:
        total_samples = args.num_steps * args.batch_size
    else:
        total_samples = len(data_layer) * args.num_epochs

    print_once("Done in {0}".format(time.time() - start_time))
    print("total samples tested: ", total_samples)
    print("total time (excluded audio processing): ", total_time, "s")
    perf = total_samples / total_time
    print("rnnt_train throughput: {:.3f} sample per second".format(perf))
    print("rnnt_train latency: ", args.batch_size / perf)
    # print_once("Final Evaluation ....................... ......  ... .. . .")
    # evalutaion()
    # WA, wait for https://jira.devtools.intel.com/browse/PYTORCHDGQ-2988 fix
    if args.distributed and args.xpu:
        torch.distributed.destroy_process_group()
    save(model, optimizer, epoch, output_dir=args.output_dir)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #assert(torch.cuda.is_available())
    #torch.backends.cudnn.benchmark = args.cudnn
    if args.batch_split_factor != 1:
        assert args.batch_size % args.batch_split_factor == 0, f'{args.batch_size} % {args.batch_split_factor} != 0'

    args.distributed = False

    mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
    print("----mpi_world_size: ", mpi_world_size)

    if mpi_world_size > 0:
        os.environ['MASTER_ADDR'] = args.dist_url #'127.0.0.1'
        os.environ['MASTER_PORT'] = args.dist_port #'29500'
        os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
        os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
        args.rank = int(os.environ.get('PMI_RANK', -1))
        args.world_size = int(os.environ.get("WORLD_SIZE", -1))
        init_method = 'tcp://' + args.dist_url + ':' + args.dist_port

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    print("----args.distributed: ", args.distributed)
    if args.dist_backend == 'ccl':
        try:
            import oneccl_bindings_for_pytorch
        except ImportError:
            print("oneccl_bindings_for_pytorch not available!")

    if args.xpu:
        if args.xpu_id is not None:
            args.xpu_id = "xpu:{}".format(args.xpu_id)

    if args.distributed:
        if args.xpu:
            torch.distributed.init_process_group(backend='ccl', init_method=init_method, world_size=args.world_size, rank=args.rank)
            args.local_rank = os.environ['MPI_LOCALRANKID']
            args.xpu_id = torch.device("xpu", int(args.local_rank))
            print('world_size:{}, rank:{}, local_rank:{}, device:{}'.format(args.world_size, args.rank, args.local_rank, args.xpu_id))
            args.N_gpu = args.world_size
            print("----N_gpu: ", args.N_gpu)
            # train_sampler = None
        else:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 1
    train_sampler = None

    multi_gpu = False
    if args.cuda:
        multi_gpu = torch.distributed.is_initialized()
        if multi_gpu:
            print_once("DISTRIBUTED TRAINING with {} gpus".format(torch.distributed.get_world_size()))
    elif args.xpu and args.distributed:
        multi_gpu = True

    # define amp optimiation level
    if args.fp16:
        optim_level = Optimization.mxprO1
    else:
        optim_level = Optimization.mxprO0

    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    ctc_vocab = add_blank_label(dataset_vocab)

    train_manifest = args.train_manifest
    val_manifest = args.val_manifest
    tst_manifest = args.tst_manifest
    featurizer_config = model_definition['input']
    featurizer_config_eval = model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level
    featurizer_config_eval["optimization_level"] = optim_level

    sampler_type = featurizer_config.get("sampler", 'default')
    perturb_config = model_definition.get('perturb', None)
    if args.pad_to_max:
        assert(args.max_duration > 0)
        featurizer_config['max_duration'] = args.max_duration
        featurizer_config_eval['max_duration'] = args.max_duration
        featurizer_config['pad_to'] = "max"
        featurizer_config_eval['pad_to'] = "max"
    print_once('model_config')
    print_dict(model_definition)

    if args.gradient_accumulation_steps < 1:
        raise ValueError('Invalid gradient accumulation steps parameter {}'.format(args.gradient_accumulation_steps))
    if args.batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError('gradient accumulation step {} is not divisible by batch size {}'.format(args.gradient_accumulation_steps, args.batch_size))


    preprocessor = preprocessing.AudioPreprocessing(**featurizer_config)
    #preprocessor.cuda()
    if args.cuda == True:
        preprocessor.cuda()
    else:
        preprocessor.cpu()

    augmentations = preprocessing.SpectrogramAugmentation(**featurizer_config)
    #augmentations.cuda()
    if args.cuda == True:
        augmentations.cuda()
    else:
        augmentations.cpu()

    train_transforms = torchvision.transforms.Compose([
        lambda xs: [x.cpu() for x in xs],
        lambda xs: [*preprocessor(xs[0:2]), *xs[2:]],
        lambda xs: [augmentations(xs[0]),   *xs[1:]],
        lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]],
    ])

    eval_transforms = torchvision.transforms.Compose([
        lambda xs: [x.cpu() for x in xs],
        lambda xs: [*preprocessor(xs[0:2]), *xs[2:]],
        lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]],
    ])

    data_layer = AudioToTextDataLayer(
                                    dataset_dir=args.dataset_dir,
                                    featurizer_config=featurizer_config,
                                    perturb_config=perturb_config,
                                    manifest_filepath=train_manifest,
                                    labels=dataset_vocab,
                                    batch_size=args.batch_size // args.gradient_accumulation_steps,
                                    multi_gpu=multi_gpu,
                                    pad_to_max=args.pad_to_max,
                                    sampler=sampler_type)

    eval_datasets = [(
        AudioToTextDataLayer(
            dataset_dir=args.dataset_dir,
            featurizer_config=featurizer_config_eval,
            manifest_filepath=val_manifest,
            labels=dataset_vocab,
            batch_size=args.eval_batch_size,
            multi_gpu=multi_gpu,
            pad_to_max=args.pad_to_max
        ),
        args.eval_frequency,
        'Eval clean',
    )]

    if tst_manifest:
        eval_datasets.append((
            AudioToTextDataLayer(
                dataset_dir=args.dataset_dir,
                featurizer_config=featurizer_config_eval,
                manifest_filepath=tst_manifest,
                labels=dataset_vocab,
                batch_size=args.eval_batch_size,
                multi_gpu=multi_gpu,
                pad_to_max=args.pad_to_max
            ),
            args.test_frequency,
            'Test other',
        ))

    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(ctc_vocab)
    )

    if args.ckpt is not None:
        print_once("loading model from {}".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        args.start_epoch = checkpoint['epoch']
    else:
        args.start_epoch = 0

    print("Calc loss...")
    loss_fn = TransducerLoss(blank_idx=len(ctc_vocab) - 1, precision=args.bf16)
    # loss_fn = RNNTLoss(len(ctc_vocab) - 1)

    N = len(data_layer)
    if sampler_type == 'default':
        args.step_per_epoch = math.ceil(N / (args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))
    elif sampler_type == 'bucket':
        args.step_per_epoch = int(len(data_layer.sampler) / args.batch_size )

    print_once('-----------------')
    print_once('Have {0} examples to train on.'.format(N))
    print_once('Have {0} steps / (gpu * epoch).'.format(args.step_per_epoch))
    print_once('-----------------')

    constant_lr_policy = lambda _: args.lr
    fn_lr_policy = constant_lr_policy
    if args.lr_decay:
        pre_decay_policy = fn_lr_policy
        fn_lr_policy = lambda s: lr_decay(args.num_epochs * args.step_per_epoch, s, pre_decay_policy(s))
    if args.lr_warmup:
        pre_warmup_policy = fn_lr_policy
        fn_lr_policy = lambda s: lr_warmup(args.lr_warmup, s, pre_warmup_policy(s) )

    if args.xpu:
        model = model.to(args.xpu_id)
    if args.bf16:
        model.bfloat16()
    # else:
    #     model.cuda()
    #    model.cpu()
    if args.distributed:
        print("-----args.xpu----", args.xpu_id)
        torch.xpu.set_device(args.xpu_id)
        if args.large_first_bucket:
            # set the first bucket with maximal size to cover all parameters for allreduce
            torch.distributed._DEFAULT_FIRST_BUCKET_BYTES = sys.maxsize
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.xpu_id],
                                                          broadcast_buffers=False if args.disable_broadcast_buffers else True,
                                                          bucket_cap_mb=args.bucket_cap,
                                                          gradient_as_bucket_view=args.use_gradient_as_bucket_view)


    if args.optimizer_kind == "novograd":
        optimizer = Novograd(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.optimizer_kind == "adam":
        optimizer = AdamW(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError("invalid optimizer choice: {}".format(args.optimizer_kind))

    # if optim_level in AmpOptimizations:
    #     model, optimizer = amp.initialize(
    #         min_loss_scale=0.125,
    #         models=model,
    #         optimizers=optimizer,
    #         opt_level=AmpOptimizations[optim_level]
    #     )

    if args.cuda and optim_level in AmpOptimizations:
        assert False, "not supported in intel_extension_for_pytorch"

    if args.ckpt is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    #model = model_multi_gpu(model, multi_gpu)
    print_once(model)
    print_once("# parameters: {}".format(sum(p.numel() for p in model.parameters())))
    greedy_decoder = RNNTGreedyDecoder(len(ctc_vocab) - 1, model.module if multi_gpu else model)


    if args.tb_path and args.local_rank == 0:
        logger = TensorBoardLogger(args.tb_path, model.module if multi_gpu else model, args.histogram)
    else:
        logger = DummyLogger()

    profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

    train(
        data_layer=data_layer,
        model=model,
        loss_fn=loss_fn,
        greedy_decoder=greedy_decoder,
        optimizer=optimizer,
        data_transforms=train_transforms,
        labels=ctc_vocab,
        optim_level=optim_level,
        multi_gpu=multi_gpu,
        fn_lr_policy=fn_lr_policy,
        evalutaion=evaluator(model, eval_transforms, loss_fn, greedy_decoder, ctc_vocab, eval_datasets, logger),
        logger=logger,
        profiling=profiling,
        args=args)

def parse_args():
    parser = argparse.ArgumentParser(description='RNNT Training Reference')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='data batch size')
    parser.add_argument("--eval_batch_size", default=1, type=int, help='eval data batch size')
    parser.add_argument("--num_epochs", default=10, type=int, help='number of training epochs. if number of steps if specified will overwrite this')
    parser.add_argument("--num_steps", default=None, type=int, help='if specified overwrites num_epochs and will only train for this number of iterations')
    parser.add_argument("--save_freq", dest="save_frequency", default=300, type=int, help='number of epochs until saving checkpoint. will save at the end of training too.')
    parser.add_argument("--eval_freq", dest="eval_frequency", default=1, type=int, help='number of epochs until doing evaluation on full dataset')
    parser.add_argument("--test_freq", dest="test_frequency", default=2, type=int, help='number of epochs until doing test on full dataset')
    parser.add_argument("--train_freq", dest="train_frequency", default=25, type=int, help='number of iterations until printing training statistics on the past iteration')
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight_decay", default=1e-3, type=float, help='weight decay rate')
    parser.add_argument("--train_manifest", type=str, required=True, help='relative path given dataset folder of training manifest file')
    parser.add_argument("--model_toml", type=str, required=True, help='relative path given dataset folder of model configuration file')
    parser.add_argument("--val_manifest", type=str, required=True, help='relative path given dataset folder of evaluation manifest file')
    parser.add_argument("--tst_manifest", type=str, required=False, help='relative path given dataset folder of test manifest file')
    parser.add_argument("--max_duration", type=float, help='maximum duration of audio samples for training and evaluation')
    parser.add_argument("--pad_to_max", action="store_true", default=False, help="pad sequence to max_duration")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help='number of accumulation steps')
    parser.add_argument("--optimizer", dest="optimizer_kind", default="novograd", type=str, help='optimizer')
    parser.add_argument("--dataset_dir", dest="dataset_dir", required=True, type=str, help='root dir of dataset')
    parser.add_argument("--lr_decay", action="store_true", default=False, help='use learning rate decay')
    parser.add_argument("--lr_warmup", type=int, default=None, help='if provided, the learning rate will linearly scale for given number of iterations from zero')
    parser.add_argument("--cudnn", action="store_true", default=False, help="enable cudnn benchmark")
    parser.add_argument("--fp16", action="store_true", default=False, help="use mixed precision training")
    parser.add_argument("--output_dir", type=str, required=True, help='saves results in this directory')
    parser.add_argument("--ckpt", default=None, type=str, help="if specified continues training from given checkpoint. Otherwise starts from beginning")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--tb_path", default=None, type=str, help='where to store tensorboard data')
    parser.add_argument("--histogram", default=False, action='store_true', help='whether to log param and grad histograms')
    parser.add_argument("--print_time", action='store_true', help='print encoder decoder time', default=False)
    parser.add_argument("--xpu", action='store_true', help='use xpu', default=False)
    parser.add_argument("--cuda", action='store_false', help='use cuda', default=False)
    parser.add_argument('--bf16', default=0, type=int, help='Datatype used: bf16')
    parser.add_argument("--warm_up", help='warm up steps, will only measure the performance from step=warm_up to step=(steps-warm_up)', type=int, default=3)
    parser.add_argument('--xpu_id', default=0, type=int, help='XPU id to use.')
    parser.add_argument('--dist_backend', type=str, default="ccl", help='Specify distributed backend to use')
    parser.add_argument('--dist_url', default='127.0.0.1', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_port', default='29500', type=str,
                        help='url port used to set up distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK',0), type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--batch_split_factor', default=1, type=int, help='Split batches feed into the joint net')

    parser.add_argument('--bucket-cap', default=25, type=int, help='controls the bucket size in MegaBytes')
    parser.add_argument('--disable-broadcast-buffers', action='store_true', help='disable syncing buffers')
    parser.add_argument('--large-first-bucket', action="store_true",
                        help='Configure a large capacity of the first bucket in DDP for allreduce')
    parser.add_argument("--use-gradient-as-bucket-view", action='store_true',
                        help="Turn ON gradient_as_bucket_view optimization in DDP")
    # store_true True
    args=parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
    print_dict(vars(args))
    main(args)
