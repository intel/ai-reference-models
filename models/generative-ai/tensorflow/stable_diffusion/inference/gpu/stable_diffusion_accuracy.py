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
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

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
    "--use_xla",
    action="store_true",
    default=False,
    help="whether to enable XLA compilation",
)
parser.add_argument(
    "--num_steps", default=50, type=int, help="number of diffusion steps"
)
parser.add_argument(
    "--precision", default="fp32", type=str, help="precision, only support(fp32, fp16)"
)
parser.add_argument(
    "--load_ref_result", default=False, action="store_true", help="whether to load reference result"
)
parser.add_argument(
    "--ref_result_dir", required=True, type=str, help="path to load/store refence result array"
)
parser.add_argument(
    "--store_result_dir",required=True, type=str, help="path to store the result image"
)
args = parser.parse_args()


def text2image(plot=False):
    model = StableDiffusion(
        img_width=512,
        img_height=512,
        jit_compile=args.use_xla,
        precision=args.precision,
    )

    prompt_lst = [
        "a photo of an astronaut riding a horse on mars",
        "hyper realistic photo of very friendly and dystopian crater",
        "ramen carved out of fractal rose ebony, in the style of hudson river school",
        "ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus",
        "a thundering retro robot crane inks on parchment with a droopy french bulldog",
        "portrait painting of a shimmering greek hero, next to a loud frill-necked lizard",
        "an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms",
    ]
    seed = 65537
    batch_size = 1
    
    if args.load_ref_result and args.ref_result_dir:
        print(f"loading reference result from {args.ref_result_dir}")
        real_img_lst = np.reshape(np.loadtxt(args.ref_result_dir, dtype=np.int_), [len(prompt_lst), 512, 512, 3])
    else:
        print("regenerating real images")        
        real_img_lst = []
        for prompt in prompt_lst:
            real_image = model.text_to_image(
                prompt=prompt,
                batch_size=batch_size,
                num_steps=args.num_steps,
                seed=seed,
            )         
            real_img_lst.append(real_image)   
        real_img_lst = np.concatenate(real_img_lst, axis=0)
        np.savetxt(args.ref_result_dir, np.reshape(real_img_lst, -1), fmt='%d')        
            
    fake_img_lst = []
    for prompt in prompt_lst:
        fake_image = model.text_to_image(
            prompt=prompt,
            batch_size=batch_size,
            num_steps=args.num_steps,
            seed=seed,
        )      
        fake_img_lst.append(fake_image)
    fake_img_lst = np.concatenate(fake_img_lst, axis=0)

    fid_score = fid(real_img_lst, fake_img_lst)
    required_fid_score = -922.2165474372248
    if np.allclose([fid_score], [required_fid_score], rtol=6, atol=1e-3):
        print(f"accuray passed, required fid score is {required_fid_score}, and actual fid score is {fid_score}")
    else:
        print(f"accuray failed, required fid score is {required_fid_score}, and actual fid score is {fid_score}")
        
    if plot:
      plot_images("gpu_real", real_img_lst)
      plot_images("gpu_fake", fake_img_lst)
    print(f"for given {len(prompt_lst)} prompts, Fid is {fid_score}")



def plot_images(comments, images):
    path = args.store_result_dir
    if not os.path.isdir(path):
        os.mkdir(path)    
    png_name = "{}/{}_{}_imgs_{}steps.png".format(path,
        comments, args.precision,  args.num_steps
    )
    print("Start plotting the generated images to %s" % (png_name))
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.savefig(png_name)



if __name__ == "__main__":
    text2image(plot=True)
