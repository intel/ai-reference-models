import time
from keras_cv.models.stable_diffusion import StableDiffusion, StableDiffusionV2
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from keras_fid import fid
import os
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


parser = argparse.ArgumentParser("Stable Diffusion inference with TensorFlow")
parser.add_argument(
    "--batch_size", default=1, type=int, help="number of images generated at one time"
)
parser.add_argument(
    "--use_xla",
    action="store_true",
    default=False,
    help="whether to enable XLA compilation",
)
parser.add_argument(
    "--num_steps", default=50, type=int, help="number of diffusion steps"
)
parser.add_argument(
    "--prompt",
    default="a photo of an astronaut riding a horse on mars",
    type=str,
    help="the text prompt list to put into the text encoder",
)
parser.add_argument(
    "--precision", default="fp32", type=str, help="precision, only support(fp32, fp16)"
)
parser.add_argument("--iterations", type=int, default=200, help="number of iterations")
args = parser.parse_args()


def text2image():
    model = StableDiffusion(
        img_width=512,
        img_height=512,
        jit_compile=args.use_xla,
        precision=args.precision,
    )
    seed = 65537
    print("Start Warmup")
    model.text_to_image(
        "warming up the model", batch_size=args.batch_size, num_steps=args.num_steps
    )
    # Start inference
    print("Start running inference and generating images")
    start = time.time()
    for i in range(args.iterations):
        images = model.text_to_image(prompt=args.prompt, batch_size=args.batch_size, seed=seed)
    end = time.time()
    latency = (end - start) / args.iterations / args.num_steps
    throughput = 1 / latency
    print("latency {} ms, throughput {} it/s".format(latency * 1000, throughput))
    return images


def plot_images(images):
    path = "./images"
    if not os.path.isdir(path):
        os.mkdir(path)
    png_name = "{}/{}_imgs_{}steps.png".format(path,
        args.precision, args.num_steps
    )
    print("Start plotting the generated images to %s" % (png_name))
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.savefig(png_name)



if __name__ == "__main__":
    images = text2image()
    plot_images(images)
