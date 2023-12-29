import os
import time
import keras_cv
from tensorflow import keras
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import json
import glob

import numpy as np
from numpy import cov, trace, iscomplexobj
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

from PIL import Image
import tensorflow as tf

class eval_stable_diffusion:
    """Evaluate Stable Diffusion model"""

    def __init__(self):

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-b', "--batch-size",
                                help='Specify the batch size. If this '
                                     'parameter is not specified, then '
                                     'it will run with batch size of 1 ',
                                dest='batch_size', type=int, default=1)

        arg_parser.add_argument("-p", "--precision",
                                help="Specify the model precision to use: fp32, bfloat16 or fp16",
                                required=True, choices=["fp32", "bfloat16", "fp16"],
                                dest="precision")
        
        arg_parser.add_argument("-d", "--data-location",
                                help='Specify the location of the data. ',
                                dest="data_location", type=str, default=None)
        
        arg_parser.add_argument('-s', "--steps", type=int, default=50,
                                help="number of steps for diffusion model")
        
        arg_parser.add_argument("-o", "--output-dir",
                                help="Specify the location of the output " + \
                                     "directory for logs and saved model",
                                dest='output_dir', required=True)

        arg_parser.add_argument('-r', "--accuracy-only",
                                help='For accuracy measurement only.',
                                dest='accuracy_only', action='store_true')

        # parse the arguments
        self.args = arg_parser.parse_args()

    def plot_images(self, images):
        plt.figure(figsize=(20, 20))
        for i in range(len(images)):
            ax = plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i])
            plt.axis("off")

    # scale an array of images to a new size
    def scale_images(self, images, new_shape):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return np.asarray(images_list)

    # calculate frechet inception distance
    def calculate_fid(self, model, images1, images2):
        # calculate activations
        act1 = model.predict(images1)
        act2 = model.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def run(self):

        print("Run inference with " + str(self.args.precision) + " precision with a batch size of " + str(self.args.batch_size))

        if self.args.precision == "bfloat16":
            print("Enabling auto-mixed precision for bfloat16")
            tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16': True})
            print(tf.config.optimizer.get_experimental_options())
        elif self.args.precision == "fp16":
            print("Enabling auto-mixed precision for fp16")
            tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
            print(tf.config.optimizer.get_experimental_options())

        model = keras_cv.models.StableDiffusionV2(img_width=512, img_height=512)

        if (not self.args.accuracy_only):
            print("Performance Benchmarking")

            print("First run for warming up the model")
            model.text_to_image("warming up the model", batch_size=self.args.batch_size)

            start = time.time()
            images = model.text_to_image(
                "a cute magical flying dog, fantasy art, "
                "golden color, high quality, highly detailed, elegant, sharp focus, "
                "concept art, character concepts, digital painting, mystery, adventure",
                batch_size=self.args.batch_size,
                num_steps=self.args.steps,
            )
            end = time.time()

            total_time = end - start
            print('Batch size = %d' % self.args.batch_size)
            if (self.args.batch_size == 1):
                print('Latency: %.3f s' % (total_time))
            # print throughput for both batch size 1 and batch_size
            print("Avg Throughput: " + str(self.args.batch_size / total_time) + " examples/sec")

            self.plot_images(images)

            print(f"Inference time: {(end - start):.2f} seconds")
            keras.backend.clear_session()
        else:  # accuracy check
            print("Accuracy Check")

            if not self.args.data_location:
                exit("Please provide a path to the COCO dataset to compute accuracy using --data-location arg.")

            with open(f'{self.args.data_location}/annotations/captions_val2017.json', 'r') as f:
                data = json.load(f)
                data = data['annotations']

            img_cap_pairs = []

            for sample in data:
                img_name = '%012d.jpg' % sample['image_id']
                img_cap_pairs.append([img_name, sample['caption']])

            captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
            captions['image'] = captions['image'].apply(
                lambda x: f'{self.args.data_location}/val2017/{x}'
            )
            captions = captions.drop_duplicates(subset=['image'])
            captions = captions.sample(10, random_state=2023)
            captions = captions.reset_index(drop=True)
            print("Dataset shape: ", captions.shape)
            print("Data: ", captions.head())

            if not os.path.isdir(os.path.join(self.args.output_dir, "original")):
                os.makedirs(os.path.join(self.args.output_dir, "original"))
            else:
                files = glob.glob(os.path.join(self.args.output_dir, "original", "*"))
                for f in files:
                    os.remove(f)
            if not os.path.isdir(os.path.join(self.args.output_dir, "generated")):
                os.makedirs(os.path.join(self.args.output_dir, "generated"))
            else:
                files = glob.glob(os.path.join(self.args.output_dir, "generated", "*"))
                for f in files:
                    os.remove(f)

            original_imgs, generated_imgs = [], []
            for index, row in captions.iterrows():
                
                img = Image.open(row["image"])
                np_img = np.asarray(img)
                np_img = np_img.reshape(1, np_img.shape[0], np_img.shape[1], np_img.shape[2])
                scaled_img = self.scale_images(np_img, (512, 512, 3))
                original_imgs.append(scaled_img)
                save_img = scaled_img.reshape(scaled_img.shape[1], scaled_img.shape[2], scaled_img.shape[3])
                im = Image.fromarray(save_img)
                im.save(self.args.output_dir + "/original/" + row["image"].split('/')[-1])

                print("Generating image with Stable Diffusion for the prompt: ", row["caption"])
                generated_img = model.text_to_image(row["caption"], seed=2023, num_steps=self.args.steps)
                generated_imgs.append(generated_img)
                save_img = generated_img.reshape(generated_img.shape[1], generated_img.shape[2], generated_img.shape[3])
                im = Image.fromarray(save_img)
                im.save(self.args.output_dir + "/generated/" + row["image"].split('/')[-1])
                print("Saved the generated image at: ", str(self.args.output_dir) + "/generated/" + row["image"].split('/')[-1])

            original_imgs_np = np.array(original_imgs)
            original_imgs_np = original_imgs_np.reshape(original_imgs_np.shape[0], original_imgs_np.shape[2], original_imgs_np.shape[3], original_imgs_np.shape[4])

            generated_imgs_np = np.array(generated_imgs)
            generated_imgs_np = generated_imgs_np.reshape(generated_imgs_np.shape[0], generated_imgs_np.shape[2], generated_imgs_np.shape[3], generated_imgs_np.shape[4])

            original_imgs_np = preprocess_input(original_imgs_np)
            generated_imgs_np = preprocess_input(generated_imgs_np)

            inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(512,512,3))
            fid = self.calculate_fid(inception_model, original_imgs_np, generated_imgs_np)
            print('FID score between original and generated images: %.3f' % fid)



if __name__ == "__main__":
    evaluate_stable_diffusion = eval_stable_diffusion()
    evaluate_stable_diffusion.run()
