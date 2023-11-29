# example of calculating the frechet inception distance in Keras
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def calculate_fid(model, images1, images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def fid(images1, images2):
    if images1.shape[0] == 1:
        images1 = np.repeat(images1, 2, axis=0)
    if images2.shape[0] == 1:
        images2 = np.repeat(images2, 2, axis=0)
    assert (images1.shape == images2.shape)
    # convert integer to floating point values
    images1 = images1.astype("float32")
    images2 = images2.astype("float32")
    H, W, C = images1.shape[1], images1.shape[2], images1.shape[3]
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # fid between images1 and images1
    model = InceptionV3(include_top=False, pooling="avg", input_shape=(H, W, C))
    fid = calculate_fid(model, images1, images2)
    return fid
