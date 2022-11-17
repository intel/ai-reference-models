import os
import torch
import time

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, flag, channels_flag, image, top_k=-1, prob_threshold=None, profiling=False, args=None, iter=0):
        cpu_device = torch.device("cpu")
        if args is not None and args.dummy > 0:
            height = args.image_size
            width = args.image_size
            images = torch.randn(args.batch_size, 3, height, height)
        else:
            height, width, _ = image.shape
            image = self.transform(image)
            images = image.unsqueeze(0)
        if flag == 1:
            images = images.to(torch.half)
        elif flag == 2:
            images = images.to(torch.bfloat16)
        if channels_flag == 1:
            images = images.to(memory_format=torch.channels_last)
        if args is not None and args.benchmark == 1:
            images = images.to(self.device)
        with torch.autograd.profiler_legacy.profile(enabled=profiling, use_xpu=True, record_shapes=False) as prof:
            with torch.inference_mode():
                self.timer.start()
                if args is None or args.benchmark == 0:
                    images = images.to(self.device)
                scores, boxes = self.net.forward(images)

                if flag > 0:
                    boxes = boxes[0].to(torch.float)
                    scores = scores[0].to(torch.float)
                else:
                    boxes = boxes[0]
                    scores = scores[0]
                if not prob_threshold:
                    prob_threshold = self.filter_threshold

                # sync for time measurement
                torch.xpu.synchronize()
                if args is not None and args.benchmark == 1:
                    print("Inference time: ", self.timer.end())

        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)

        post_start_time = time.time()
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height

        ret = picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
        post_end_time = time.time()
        print("Post time: ", post_end_time - post_start_time)
        if args is None or args.benchmark == 0:
            print("Inference time: ", self.timer.end())

        if profiling and iter==args.profile_iter:
            title = "/ssd_mobilenetv1_inference_"
            if args.channels_last:
                title += "channels_last_"
            else:
                title += "block_"
            if args.bf16:
                title += "bf16_"
            if args.fp16:
                title += "fp16_"
            if args.int8:
                title += "int8_"
            if args.batch_size:
                title += "bs" + str(args.batch_size) + "_"

            profiling_path = os.getenv('PROFILE_PATH')
            if not profiling_path:
                profiling_path = './'
            torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), profiling_path + title + 'profiling.pt')
            torch.save(prof.key_averages(group_by_input_shape=True).table(), profiling_path + title + 'profiling_detailed.pt')
            prof.export_chrome_trace(profiling_path + title + 'profiling.json')
            print(prof.key_averages().table(sort_by="self_xpu_time_total"))
            print(prof.key_averages(group_by_input_shape=True).table())
        return ret
