import os
import torch
import torch.nn as nn
import intel_extension_for_pytorch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.data_preprocessing import PredictionTransform
from vision.ssd.config import mobilenetv1_ssd_config as config
from torch.quantization.quantize_jit import (
convert_jit,
prepare_jit,
)
from torch.quantization import default_qconfig
from torch.jit._recursive import wrap_cpp_module
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, \
                          mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. \
                          Currently support voc and open_images.')
parser.add_argument("--dataset", type=str,
    help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=False)
parser.add_argument("--use_xpu", type=str2bool, default=True)
parser.add_argument('--fp16', action='store_true', help='Datatype used: fp16')
parser.add_argument('--bf16', action='store_true', help='Datatype used: bf16')
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument('--iter_num', default=4951, type=int,
                    help='number of iteration')
parser.add_argument("--iou_threshold", type=float, default=0.5,
                    help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str,
                    help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument("--int8", action='store_true', help='use int8', default=False)
parser.add_argument('--mc', default=False,
                    help='do calibration with weight per channel quantization')
parser.add_argument('--channels_last', action='store_true',
                    help='Dataformat used: channel_last(plain NHWC)')
parser.add_argument('--calib_num', default=8, type=int,
                    help='number of calibration iteration')
parser.add_argument('--dummy', default=0, type=int,
                    help='dummy for perf')
parser.add_argument('--batch_size', default=1, type=int,
                    help='batch_size for dummy')
parser.add_argument('--image_size', default=300, type=int,
                    help='image_size for dummy')
parser.add_argument('--benchmark', default=0, type=int, help='for benchmark performance, move H2D out of end2end calculation')
parser.add_argument('--profile_iter', default=5, type=int, help='profile iter')
parser.add_argument('--num-iterations', default=100, type=int, help='iterations for benchmark test')
args = parser.parse_args()

if args.use_xpu:
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(
                    class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index] = {}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(
                all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(
                all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes,
                                        difficult_cases, prediction_file,
                                        iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(
            precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)

def jit(model, dataset):
    print("run JIT path...")

    model.eval()

    transform = PredictionTransform(config.image_size, mean=config.image_mean, std=config.image_std)
    image = dataset.get_image(0)
    height, width, _ = image.shape
    image = transform(image)
    images = image.unsqueeze(0)
    images = images.to("xpu")

    if args.fp16:
        images = images.half()

    if args.benchmark == 1:
        modelJit = torch.jit.trace(model, images, check_trace=False)
    else:
        modelJit = torch.jit.trace(model, images)

    model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(modelJit._c))
    return model

def calib(model, dataset):
    print("Calibration for INT8 ... ")
    with torch.no_grad():
        if args.mc:
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.observer.MinMaxObserver.with_args(
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                    dtype=torch.quint8
                ),
                weight=torch.quantization.default_per_channel_weight_observer
            )
        else:
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.observer.MinMaxObserver.with_args(
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                    dtype=torch.quint8
                ),
                weight=torch.quantization.default_weight_observer
            )

        model = prepare_jit(model, {'': qconfig}, True)

        # tune acc through iteration number for calibration
        for i in range(args.calib_num):
            image = dataset.get_image(i)
            height, width, _ = image.shape
            transform = PredictionTransform(config.image_size, mean=config.image_mean, std=config.image_std)
            image = transform(image)
            images = image.unsqueeze(0)
            images = images.to("xpu")
            model = model.to("xpu")
            images = images.to("xpu")

            model(images)

        model = convert_jit(model, True)

    return model



if __name__ == '__main__':
    pathlib.Path(args.eval_dir).mkdir(exist_ok=True)
    eval_path = pathlib.Path(args.eval_dir + "/" + str(os.getpid()))
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, args.iter_num, is_test=True)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(
            args.dataset, args.iter_num, dataset_type="test")

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(
        dataset)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(
            len(class_names), is_test=True, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(
            len(class_names),
            width_mult=args.mb2_width_mult,
            is_test=True)
    else:
        logging.fatal(
            "The net type is wrong. \
             It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    net.load(args.trained_model)

    # convert the model to half exclude BatchNorm2d layer
    # typeFlag 0 Float; typeFlag 1 Half; typeFlag 2 BFloat16
    typeFlag = 0
    if args.fp16 or args.bf16:
        if args.fp16:
            net.half()
            typeFlag = 1
        else:
            net.bfloat16()
            typeFlag = 2
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net = net.to(DEVICE)

    if args.fp16 or args.int8:
        net = jit(net, dataset)
    if args.int8:
        net = calib(net, dataset)

    channelsFlag = 0
    # channelsFlag 0 NCHW; channelsFlag 1 NHWC
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)
        channelsFlag = 1
    print("Running on", DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(
            net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(
            net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(
            net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(
            net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(
            net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal(
            "The net type is wrong. \
             It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    results = []
    for i in range(max(len(dataset), args.num_iterations)):
        if args.benchmark and i > args.num_iterations:
            break
        print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i) if args.dummy == 0 else None
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")

        profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        with torch.inference_mode():
            boxes, labels, probs = predictor.predict(typeFlag, channelsFlag, image, -1, None, profiling, args, i)
            if profiling and i == args.profile_iter:
                break
        # sync for time measurement
        torch.xpu.synchronize()
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))

    if args.dummy > 0:
        sys.exit(0)

    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
