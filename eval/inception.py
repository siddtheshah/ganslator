import numpy as np
import tensorflow as tf
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
import os
import glob

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_score(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def score(model_results_dir):
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))
    # define two fake collections of images
    images_real_files = glob.glob(os.path.join(model_results_dir, "real*"))
    images_fake_files = glob.glob(os.path.join(model_results_dir, "fake*"))

    images_real = [Image.open(file) for file in images_real_files]
    images_fake = [Image.open(file) for file in images_fake_files]

    real = np.stack(images_real)
    fake = np.stack(images_fake)

    print('Prepared', real.shape, fake.shape)
    # convert integer to floating point values
    images1 = real.astype('float32')
    images2 = fake.astype('float32')
    # resize images
    images1 = scale_images(images1, (256, 256, 3))
    images2 = scale_images(images2, (256, 256, 3))
    print('Scaled', images1.shape, images2.shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # fid between images1 and images1
    fid_id = calculate_score(model, images1, images1)
    print('FID (same): %.3f' % fid_id)
    # fid between images1 and images2
    fid_cross = calculate_score(model, images1, images2)
    print('FID (different): %.3f' % fid_cross)

    return fid_id, fid_cross
