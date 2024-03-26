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
import multiprocessing as mp
import time
import torch
import sys
import torch._C
import torch.jit._recursive

# sample modules
import arguments_utils
import barrier_utils
import configure_utils
import io_utils
import loader_utils
import statistics_utils
import summary_utils

from arguments_utils import args
from pathlib import Path
from torch.jit._recursive import wrap_cpp_module

try:
    from yolov5.utils.plots import Annotator, save_one_box, colors
    from yolov5.utils.general import non_max_suppression, increment_path, scale_coords
except Exception as e:
    print('fatal: ' + str(e), file=sys.stderr)
    print('fatal: download yolov5 model sources to yolov5 folder', file=sys.stderr)
    print('fatal: or set PYTHONPATH to the location of yolov5 folder', file=sys.stderr)
    sys.exit(1)

ROOT = Path(os.getcwd()) 

conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
line_thickness=3  # bounding box thickness (pixels)
project=ROOT / 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
save_txt=False  # save results to *.txt
hide_labels=False  # hide labels
hide_conf=False  # hide confidences

def quantize_model(model):
    if args.load:
        # loaded model should already be quantized
        return model

    # load data set for quantization
    validation_loader_calib = loader_utils.load_validation_dataset(
        args.calib_bs,
        args.calib_iters * args.calib_bs,
        args.width,
        args.height,
        loader_utils.validate_data_src(args.data, args.dummy),
        data_workers=4,
        pin_memory_device='xpu' if args.xpu else 'cuda' if args.gpu else None
    )

    io_utils.write_info('Doing JIT model quantization to integer datatype')
    from torch.jit._recursive import wrap_cpp_module
    from torch.quantization.quantize_jit import (
        convert_jit,
        prepare_jit,
    )

    # Convert model to either jit.trace or jit.script.
    # this can have an impact on both perf and quantizability.
    # If quantization fails for you try changing JIT methodology.
    if args.jit_trace:
        trace_input = torch.randn(args.calib_bs, 3, args.width, args.height)
        if args.channels_last:
            trace_input = trace_input.to(memory_format=torch.channels_last)
        if args.xpu or args.gpu:
            trace_input = trace_input.to(args.device)
        io_utils.write_info('Using JIT trace for quantization')
        with torch.inference_mode():
            model = torch.jit.trace(model, trace_input)
    elif args.jit_script:
        io_utils.write_error('Must use jit_trace for Yolov5!')
        sys.exit(1)
    else:
        io_utils.write_error('Quantization is only support for JIT models!')
        sys.exit(1)
    model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))

    # Define various parameters for quantization. Majority of these can be controled through CL arguments.
    quantization_dtype = torch.quint8
    if args.int8:
        quantization_dtype = torch.qint8
    io_utils.write_info('Quantization datatype: {0}'.format(quantization_dtype))
    if args.asymmetric_quantization:
        quantization_scheme = torch.per_tensor_symmetric
    else:
        quantization_scheme = torch.per_tensor_affine
    io_utils.write_info('Quantization scheme: {0}'.format(quantization_scheme))
    if args.perchannel_weight:
        quantization_weight_observer = torch.quantization.default_per_channel_weight_observer
    else:
        quantization_weight_observer = torch.quantization.default_weight_observer
    io_utils.write_info('Quantization weight observer: {0}'.format(quantization_weight_observer))
    quantization_activation = torch.quantization.observer.MinMaxObserver
    io_utils.write_info('Quantization activation: {0}'.format(quantization_activation))

    # Generate quantization config
    quantization_config = torch.quantization.QConfig(
        activation=quantization_activation.with_args(
            qscheme=quantization_scheme,
            reduce_range=False,
            dtype=quantization_dtype
        ),
        weight=quantization_weight_observer
    )

    io_utils.write_info('Full quantization config: {0}'.format(quantization_config))

    # Run sample data through model to do quantization
    with torch.inference_mode():
        io_utils.write_info('Preparing JIT model for quantization')
        model = prepare_jit(model, {'': quantization_config}, True)

        io_utils.write_info('Starting quantization loop with {0} batches of size {1}'.format(args.calib_iters, args.calib_bs))
        for batch_index, (images, target) in enumerate(validation_loader_calib):
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            if args.xpu or args.gpu:
                images = images.to(args.device, non_blocking=args.non_blocking)
            model(images)
            
            if args.xpu:
                torch.xpu.synchronize(args.device)
            elif args.gpu:
                torch.cuda.synchronize(args.device)
        io_utils.write_info('Quantization loop completed')

        io_utils.write_info('Converting JIT quantization model to standard JIT model')
        model = convert_jit(model, True)

    io_utils.write_info('Quantization completed')
    return model

def do_warmup_acc(model, ds):
    ds_batches = len(ds)

    # Final config of the model
    use_autocast, autocast_dtype, model = inference_config(model)

    # Main perf testing loop.
    io_utils.write_info('Starting warmup')
    with torch.inference_mode():
        if args.channels_last:
            io_utils.write_info('Images will be converted to channels last format')
        for path, images, im0s, vid_cap, s in ds:
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)

            if args.xpu:
                try:
                    import memory_check
                    memory_check.display_mem(args.device)
                except:
                    pass
                images = torch.from_numpy(images).to(args.device)
                if args.no_amp:
                    if args.fp16:
                        images = images.to(dtype=torch.float16)
                    elif args.bf16:
                        images = images.to(dtype=torch.bfloat16)
                images = images / 255 # 0 - 255 to 0.0 - 1.0
                if len(images.shape) == 3:
                    images = images[None]  # expand for batch dim
                if args.jit_trace:
                    # warmup
                    for batch_repeat_index in range(min([args.batch_streaming, args.warm_up])):
                        pred = model(images)
                        torch.xpu.synchronize(args.device)                                                            
                        # NMS
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                        pred = [x.to("cpu") for x in pred]                       
                else:
                    io_utils.write_error('Must use jit_trace for Yolov5!')
                    sys.exit(1)
            else:
                if args.gpu:
                    try:
                        import memory_check
                        memory_check.display_mem(args.device)
                    except:
                        pass
                    images = images.cuda(args.device, non_blocking=args.non_blocking)
                if args.no_amp:
                    if args.fp16:
                        images = images.to(dtype=torch.float16)
                    elif args.bf16:
                        images = images.to(dtype=torch.bfloat16)
                if args.jit_trace:
                    # warmup
                    for batch_repeat_index in range(min([args.batch_streaming, args.warm_up])):
                        pred = model(images)
                        torch.cuda.synchronize(args.device)                                                            
                        # NMS
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                        pred = [x.to("cpu") for x in pred]
                else:
                    io_utils.write_error('Must use jit_trace for Yolov5!')
                    sys.exit(1)

    io_utils.write_info('Completed {0} warmup batches'.format(args.warm_up))
    return use_autocast, autocast_dtype, model

def do_warmup(model, ds):
    ds_batches = len(ds)

    # Final config of the model
    use_autocast, autocast_dtype, model = inference_config(model)

    # Main perf testing loop.
    io_utils.write_info('Starting warmup')

    with torch.inference_mode():
        if args.channels_last:
            io_utils.write_info('Images will be converted to channels last format')
        for batch_index, (images, target) in enumerate(ds):
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)

            if args.xpu:
                try:
                    import memory_check
                    memory_check.display_mem(args.device)
                except:
                    pass
                images = images.to(args.device, non_blocking=args.non_blocking)
                if args.no_amp:
                    if args.fp16:
                        images = images.to(dtype=torch.float16)
                    elif args.bf16:
                        images = images.to(dtype=torch.bfloat16)

                if args.jit_trace:
                    # warmup
                    for batch_repeat_index in range(min([args.batch_streaming, args.warm_up])): 
                        pred = model(images)
                        torch.xpu.synchronize(args.device)                                                            
                        # NMS
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                        pred = [x.to("cpu") for x in pred]                       
                else:
                    io_utils.write_error('Must use jit_trace for Yolov5!')
                    sys.exit(1)
            else:
                if args.gpu:
                    try:
                        import memory_check
                        memory_check.display_mem(args.device)
                    except:
                        pass
                    images = images.cuda(args.device, non_blocking=args.non_blocking)
                if args.no_amp:
                    if args.fp16:
                        images = images.to(dtype=torch.float16)
                    elif args.bf16:
                        images = images.to(dtype=torch.bfloat16)
                if args.jit_trace:
                    # warmup
                    for batch_repeat_index in range(min([args.batch_streaming, args.warm_up])):
                        pred = model(images)
                        torch.cuda.synchronize(args.device)                                                            
                        # NMS
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                        pred = [x.to("cpu") for x in pred]
                else:
                    io_utils.write_error('Must use jit_trace for Yolov5!')
                    sys.exit(1)

    io_utils.write_info('Completed {0} warmup batches'.format(args.warm_up))
    return use_autocast, autocast_dtype, model

def inference_config(model):
    use_autocast = False
    if not args.no_amp:
        if args.bf16 or args.fp16:
            io_utils.write_info('Using autocast')
            use_autocast = True
    else:
        if args.bf16 or args.fp16:
            io_utils.write_info('Not using autocast')

    model.trainable = False
    io_utils.write_info('Model training disabled')

    # switch to evaluate mode
    model.eval()
        
    autocast_dtype = torch.float32
    if args.fp16:
        autocast_dtype = torch.float16
    elif args.bf16:
        autocast_dtype = torch.bfloat16

    if args.no_amp:
        if args.fp16:
            model = model.to(dtype=torch.float16)
        elif args.bf16:
            model = model.to(dtype=torch.bfloat16)

    if args.no_grad:
        model.zero_grad(set_to_none=True)
        io_utils.write_info('Using no gradient')
    elif args.zero_grad:
        model.zero_grad()
        io_utils.write_info('Using zero gradient')

    if not any([args.int8, args.uint8]):
        if args.jit_trace and not args.load:
            trace_input = torch.randn(args.batch_size, 3, args.width, args.height)
            if args.channels_last:
                trace_input = trace_input.to(memory_format=torch.channels_last)
            if args.xpu or args.gpu:
                trace_input = trace_input.to(args.device)
            if args.no_amp and any([args.fp16, args.bf16]):
                trace_input = trace_input.to(dtype=autocast_dtype)
            io_utils.write_info('Using JIT trace')
            with torch.inference_mode():
                if args.xpu:
                    with torch.xpu.amp.autocast(enabled=use_autocast, dtype=autocast_dtype, cache_enabled=False):
                        model = torch.jit.trace(model, trace_input, check_trace=False)
                    model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(model._c))
                elif args.gpu:
                    with torch.autocast(enabled=use_autocast, device_type='cuda' if args.gpu else 'cpu', dtype=autocast_dtype, cache_enabled=False):
                        model = torch.jit.trace(model, trace_input)
        elif args.jit_script and not args.load:
            io_utils.write_error('Must use jit_trace for Yolov5!')
            sys.exit(1)

    return use_autocast, autocast_dtype, model
  
def do_perf_benchmarking(model, ds):
    ds_batches = len(ds)
    total_batches = ds_batches * args.batch_streaming
    print_frequency = max([1, total_batches // args.status_prints])

    # Profiling
    profiling = os.environ.get('PROFILE', 'OFF').upper() in ['1', 'Y', 'ON', 'YES', 'TRUE']
    if profiling:
        io_utils.write_info('Using profiling')

    # Final config of the model
    use_autocast, autocast_dtype, model = inference_config(model)

    # Safe the model before doing perf
    if args.save:
        loader_utils.save_model_to_file(model)
    
    # Create progress meter for output to terminal
    throughput = statistics_utils.average_meter('Throughput', ':.2f', ' img/s')
    latency = statistics_utils.average_meter('Latency', ':.2f', ' ms')
    top1 = statistics_utils.average_meter('Acc@1', ':.2f', '%')
    throughput_overhead = statistics_utils.average_meter('Throughput /w Overhead', ':.2f', ' img/s')
    latency_overhead = statistics_utils.average_meter('Latency /w Overhead', ':.2f', ' ms')
    progress = statistics_utils.progress_meter(
        total_batches,
        [
            throughput,
            latency,
            top1,
            throughput_overhead,
            latency_overhead
        ],
        prefix='INFO[{0}/{1}]: PERF_STATUS'.format(args.instance, args.total_instances)
    )

    # Main perf testing loop.
    #barrier_utils.do_ipc_sync(args.barrier, 'start_perf_benchmark', args.terminate_if_sync_fail)
    should_display = False
    total_duration = 0
    io_utils.write_info('Starting inference perf testing on {0} batches with {1} unique batches...'.format(total_batches, ds_batches))
    if args.dummy:
        io_utils.write_warning('Since dummy data is being used expect accuracy to be close to zero.')
    start_benchmark_time = time.time()
    with torch.inference_mode():       
        if args.channels_last:
            io_utils.write_info('Images will be converted to channels last format')      
        
        if profiling:
            profile_name = 'fp32'
            if args.fp16:
                profile_name = 'fp16'
            elif args.bf16:
                profile_name = 'bf16'

        for batch_index, (images, target) in enumerate(ds):
            total = 0
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            if args.xpu:
                with torch.autograd.profiler_legacy.profile(enabled=profiling, use_xpu=True, record_shapes=False) as prof:
                    
                    try:
                        import memory_check
                        memory_check.display_mem(args.device)
                    except:
                        pass
                    
                    images = images.to(args.device, non_blocking=args.non_blocking)
                    if args.no_amp:
                        if args.fp16:
                            images = images.to(dtype=torch.float16)
                        elif args.bf16:
                            images = images.to(dtype=torch.bfloat16)

                    if args.jit_trace:
                        # inference
                        for batch_repeat_index in range(args.batch_streaming):                      
                            torch.xpu.synchronize(args.device)
                            start_time = time.time()

                            pred = model(images)

                            torch.xpu.synchronize(args.device)
                            end_time = time.time()
                    
                            # NMS
                            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                            pred = [x.to("cpu") for x in pred]

                            duration = end_time - start_time
                            print("Inference: {:>5}  E2E Time: {:>25}     Instance: {} Current Time: {}".format(batch_repeat_index,duration, args.instance, time.time()))                           
                            total += duration
                    else:
                        io_utils.write_error('Must use jit_trace for Yolov5!')
                        sys.exit(1)

                if profiling:
                    torch.save(prof.key_averages().table(sort_by='self_xpu_time_total'), './profiling.' + profile_name + '.inf.pt')
                    torch.save(prof.table(sort_by='id', row_limit=100000), './profiling.' + profile_name + '.inf.detailed.pt')
            else:
                activities = None
                prof_sort = None
                if profiling:
                    prof_sort = 'self_cpu_time_total'
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    if args.gpu:
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                        prof_sort = 'self_cuda_time_total'
                with torch.autograd.profiler.profile(enabled=profiling, use_cuda=True if args.gpu else False, record_shapes=False) as prof:
                    if args.gpu:
                        try:
                            import memory_check
                            memory_check.display_mem(args.device)
                        except:
                            pass
                        images = images.cuda(args.device, non_blocking=args.non_blocking)
                    
                    if args.no_amp:
                        if args.fp16:
                            images = images.to(dtype=torch.float16)
                        elif args.bf16:
                            images = images.to(dtype=torch.bfloat16)

                    if args.jit_trace:
                        # inference
                        for batch_repeat_index in range(args.batch_streaming):
                            torch.cuda.synchronize(args.device)
                            start_time = time.time()
                            
                            pred = model(images)
                            
                            torch.cuda.synchronize(args.device)
                            end_time = time.time()
                            
                            # NMS
                            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                            pred = [x.to("cpu") for x in pred]

                            duration = end_time - start_time
                            print("Inference: {:>5}  E2E Time: {:>25}     Instance: {} Current Time: {}".format(batch_repeat_index,duration, args.instance, time.time()))                           
                            total += duration
                    else:
                        io_utils.write_error('Must use jit_trace for Yolov5!')
                        sys.exit(1)

                if profiling:
                    torch.save(prof.key_averages().table(sort_by=prof_sort), './profiling.' + profile_name + '.inf.pt')
                    torch.save(prof.table(sort_by='id', row_limit=100000), './profiling.' + profile_name + '.inf.detailed.pt')

            duration_eval = total / args.batch_streaming
            total_duration += duration_eval * args.batch_streaming * 1000 # milliseconds
            end_benchmark_time = time.time()
            benchmark_wall_clock_time = end_benchmark_time - start_benchmark_time

            latency.update(
                duration_eval * 1000,
                n=args.batch_streaming
                ) # milliseconds
            throughput.update(
                args.batch_size / duration_eval,
                n=args.batch_streaming
                ) # IMG/S
            latency_overhead.reset()
            latency_overhead.update(
                benchmark_wall_clock_time * 1000 / ((batch_index * args.batch_streaming) + args.batch_streaming),
                n=((batch_index * args.batch_streaming) + args.batch_streaming)
                ) # milliseconds
            throughput_overhead.reset()
            throughput_overhead.update(
                args.batch_size * ((batch_index * args.batch_streaming) + args.batch_streaming) / benchmark_wall_clock_time,
                n=((batch_index * args.batch_streaming) + args.batch_streaming)
                ) # IMG/S
            should_display = False
            for batch_repeat_index in range(args.batch_streaming):
                acc1 = [0]
                # record stats
                top1.update(acc1[0])

                if ((batch_index * args.batch_streaming) + batch_repeat_index) % print_frequency == print_frequency - 1:
                    should_display = True
                elif ((batch_index * args.batch_streaming) + batch_repeat_index) + 1 == total_batches:
                    should_display = True

            if should_display:
                progress.display((batch_index * args.batch_streaming) + args.batch_streaming)

    # If we haven't already displayed the latest data
    if not should_display:
        progress.display(total_batches)

    return total_batches, throughput.avg, latency.avg, top1.avg, throughput_overhead.avg, latency_overhead.avg

def accuracy(pred, ds, path, im0s, images, s, names):
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    acc = 0
    
    for i, det in enumerate(pred):  # per image
        p, im0, frame = path, im0s.copy(), getattr(ds, 'frame', 0)
        p = Path(p)  # to Path
        s += '%gx%g ' % images.shape[2:]  # print string
        imc = im0.copy()
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(images.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            max_person_conf = 0
            for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    if label[:6] == "person":
                        max_person_conf = max(max_person_conf, float(label[7:]))
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            acc = max_person_conf
            
    return acc

def do_accuracy_test(model, ds, names):
    ds_batches = len(ds)
    total_batches = args.batch_streaming
    print_frequency = max([1, total_batches // args.status_prints])

    # Profiling
    profiling = os.environ.get('PROFILE', 'OFF').upper() in ['1', 'Y', 'ON', 'YES', 'TRUE']
    if profiling:
        io_utils.write_info('Using profiling')

    # Final config of the model
    use_autocast, autocast_dtype, model = inference_config(model)
       
    # Safe the model before doing perf
    if args.save:
        loader_utils.save_model_to_file(model)
    
    # Create progress meter for output to terminal
    throughput = statistics_utils.average_meter('Throughput', ':.2f', ' img/s')
    latency = statistics_utils.average_meter('Latency', ':.2f', ' ms')
    top1 = statistics_utils.average_meter('Accuracy', ':.2f', '')
    throughput_overhead = statistics_utils.average_meter('Throughput /w Overhead', ':.2f', ' img/s')
    latency_overhead = statistics_utils.average_meter('Latency /w Overhead', ':.2f', ' ms')
    progress = statistics_utils.progress_meter(
        total_batches,
        [
            throughput,
            latency,
            top1,
            throughput_overhead,
            latency_overhead
        ],
        prefix='INFO[{0}/{1}]: PERF_STATUS'.format(args.instance, args.total_instances)
    )

    # Main perf testing loop.
    #barrier_utils.do_ipc_sync(args.barrier, 'start_perf_benchmark', args.terminate_if_sync_fail)
    should_display = False
    total_duration = 0
    io_utils.write_info('Starting inference perf testing on {0} batches with {1} unique batches...'.format(total_batches, ds_batches))

    start_benchmark_time = time.time()
    with torch.inference_mode():
        if args.channels_last:
            io_utils.write_info('Images will be converted to channels last format')

        if profiling:
            profile_name = 'fp32'
            if args.fp16:
                profile_name = 'fp16'
            elif args.bf16:
                profile_name = 'bf16'
        
        batch_index=0
        for path, images, im0s, vid_cap, s in ds:
            total = 0
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)

            images = torch.from_numpy(images).to(args.device)
            if args.no_amp:
                if args.fp16:
                    images = images.to(dtype=torch.float16)
                elif args.bf16:
                    images = images.to(dtype=torch.bfloat16)
                else:
                    images = images.to(dtype=torch.float32)
            images = images / 255  # 0 - 255 to 0.0 - 1.0
            if len(images.shape) == 3:
                images = images[None]  # expand for batch dim
            
            if args.xpu:
                with torch.autograd.profiler_legacy.profile(enabled=profiling, use_xpu=True, record_shapes=False) as prof:
                    try:
                        import memory_check
                        memory_check.display_mem(args.device)
                    except:
                        pass

                    images = images.to(args.device, non_blocking=args.non_blocking)
                    if args.fp16:
                      images = images.to(dtype=torch.float16)
                    if args.jit_trace:
                        # inference
                        for batch_repeat_index in range(args.batch_streaming):
                            torch.xpu.synchronize(args.device)
                            start_time = time.time()
                            
                            pred = model(images)
                            
                            torch.xpu.synchronize(args.device)
                            end_time = time.time()
                            
                            # NMS
                            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                            pred = [x.to("cpu") for x in pred]

                            duration = end_time - start_time                           
                            print("Inference: {:>5}  E2E Time: {:>25}     Instance: {} Current Time: {}".format(batch_repeat_index,duration, args.instance, time.time()))
                            total += duration
                    else:
                        io_utils.write_error('Must use jit_trace for Yolov5!')
                        sys.exit(1)

                if profiling:
                    torch.save(prof.key_averages().table(sort_by='self_xpu_time_total'), './profiling.' + profile_name + '.inf.pt')
                    torch.save(prof.table(sort_by='id', row_limit=100000), './profiling.' + profile_name + '.inf.detailed.pt')
            else:
                activities = None
                prof_sort = None
                if profiling:
                    prof_sort = 'self_cpu_time_total'
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    if args.gpu:
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                        prof_sort = 'self_cuda_time_total'
                with torch.autograd.profiler.profile(enabled=profiling, use_cuda=True if args.gpu else False, record_shapes=False) as prof:
                    if args.gpu:
                        try:
                            import memory_check
                            memory_check.display_mem(args.device)
                        except:
                            pass
                        images = images.cuda(args.device, non_blocking=args.non_blocking)
                    if args.no_amp:
                        if args.fp16:
                            images = images.to(dtype=torch.float16)
                        elif args.bf16:
                            images = images.to(dtype=torch.bfloat16)

                    if args.jit_trace:
                        # inference
                        for batch_repeat_index in range(args.batch_streaming):
                            torch.cuda.synchronize(args.device)
                            start_time = time.time()
                            
                            pred = model(images)
                            
                            torch.cuda.synchronize(args.device)
                            end_time = time.time()
                            
                            # NMS
                            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                            pred = [x.to("cpu") for x in pred]

                            duration = end_time - start_time                           
                            print("Inference: {:>5}  E2E Time: {:>25}     Instance: {} Current Time: {}".format(batch_repeat_index,duration, args.instance, time.time()))
                            total += duration
                    else:
                        io_utils.write_error('Must use jit_trace for Yolov5!')
                        sys.exit(1)

                if profiling:
                    torch.save(prof.key_averages().table(sort_by=prof_sort), './profiling.' + profile_name + '.inf.pt')
                    torch.save(prof.table(sort_by='id', row_limit=100000), './profiling.' + profile_name + '.inf.detailed.pt')
            
            duration_eval = total / args.batch_streaming
            total_duration += duration_eval * args.batch_streaming * 1000 # milliseconds
            end_benchmark_time = time.time()
            benchmark_wall_clock_time = end_benchmark_time - start_benchmark_time

            latency.update(
                duration_eval * 1000,
                n=args.batch_streaming
                ) # milliseconds
            throughput.update(
                args.batch_size / duration_eval,
                n=args.batch_streaming
                ) # IMG/S
            latency_overhead.reset()
            latency_overhead.update(
                benchmark_wall_clock_time * 1000 / ((batch_index * args.batch_streaming) + args.batch_streaming),
                n=((batch_index * args.batch_streaming) + args.batch_streaming)
                ) # milliseconds
            throughput_overhead.reset()
            throughput_overhead.update(
                args.batch_size * ((batch_index * args.batch_streaming) + args.batch_streaming) / benchmark_wall_clock_time,
                n=((batch_index * args.batch_streaming) + args.batch_streaming)
                ) # IMG/S
            should_display = False
                             
            # record stats
            top1.update(accuracy(pred, ds, path, im0s, images, s, names))

            if ((batch_index * args.batch_streaming) + batch_repeat_index) % print_frequency == print_frequency - 1:
                should_display = True
            elif ((batch_index * args.batch_streaming) + batch_repeat_index) + 1 == total_batches:
                should_display = True

            if should_display:
                progress.display((batch_index * args.batch_streaming) + args.batch_streaming)
            
            batch_index +=1

    # If we haven't already displayed the latest data
    if not should_display:
        progress.display(total_batches)

    return total_batches, throughput.avg, latency.avg, top1.avg, throughput_overhead.avg, latency_overhead.avg

def predict(instance, input_args):
    # Setup passed in args and make them global
    args = input_args
    args.instance = instance

    # Enumerate list of devices and check compatibility
    configure_utils.enum_device()

    # Enumerate dtypes to be used
    configure_utils.enum_dtypes()

    # Set environment variables to enable optimizations
    configure_utils.set_environ()

    # Apply various device configurations such as auto mixed precision, eager mode, etc.
    configure_utils.config_devices()

    # Show test conditions
    summary_utils.show_test_conditions()

    # create model

    model = loader_utils.load_model()
    names = model.names

    # Data loading code
    validation_loader_inf = loader_utils.load_validation_dataset(
        args.batch_size,
        args.max_val_dataset_size,
        args.width,
        args.height,
        model,
        loader_utils.validate_data_src(args.data, args.dummy),
        data_workers=4,
        pin_memory_device='xpu' if args.xpu else 'cuda' if args.gpu else None
    )
    
    if any([args.int8, args.uint8]):
        # Quantize model to requested integer datatype
        model = quantize_model(model)
    
    if args.dummy:
        # Do warmup on the model
        use_autocast, autocast_dtype, model = do_warmup(model, validation_loader_inf)
        # Do inference benchmarking
        
        batches_tested, throughput, latency, top1, throughput_overhead, latency_overhead = do_perf_benchmarking(model, validation_loader_inf)
        summary_utils.write_results(batches_tested, throughput, latency, top1, throughput_overhead, latency_overhead)
    else:
        # Do warmup on the model
        use_autocast, autocast_dtype, model = do_warmup_acc(model, validation_loader_inf)
        
        # Do accuracy testing
        batches_tested, throughput, latency, top1, throughput_overhead, latency_overhead = do_accuracy_test(model, validation_loader_inf, names)
        summary_utils.write_results(batches_tested, throughput, latency, top1, throughput_overhead, latency_overhead)

def main():
    arguments_utils.parse_arguments() 
    args.barrier = mp.Barrier(args.total_instances, timeout=args.max_wait_for_sync)   
    try:
        mp.set_start_method('spawn')
    except:
        pass 
    processes = {}
    for instance in range(1, args.total_instances + 1):
        processes[instance] = mp.Process(target=predict, args=(instance, args))
    for instance in processes:
        processes[instance].start()
    has_error = False
    for instance in processes:
        processes[instance].join()
        if processes[instance].exitcode != 0:
            io_utils.write_error('Process {0} returned non-zero exit code: {1}'.format(instance, processes[instance].exitcode))
            has_error = True

    summary_utils.combine_results()
    sys.exit(1 if has_error else 0)

if __name__ == '__main__':
    main()
