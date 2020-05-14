import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.applications.vgg19.preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
import os
import glob

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize
        new_image = resize(image, new_shape)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def calculate_inception_score(model, images):
    preds = model.predict(images)
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
    kl = np.mean(np.sum(kl, 1))

    return np.exp(kl)

# calculate frechet inception distance
def calculate_frechet_distance(model, images1, images2):
    # calculate activations
    m1 = model.predict(images1)
    m2 = model.predict(images2)
    print(np.shape(m1))
    act1 = np.squeeze(m1)
    act2 = np.squeeze(m2)
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
    fd = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fd

def score(model_results_dir, max_images=200):
    # prepare the inception v3 model
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))
    vgg = VGG19(include_top=True, input_shape=(224, 224, 3))
    # define two fake collections of images
    images_real_files = glob.glob(os.path.join(model_results_dir, "real*"))[:max_images]
    images_fake_files = glob.glob(os.path.join(model_results_dir, "fake*"))[:max_images]

    images_real = [Image.open(file) for file in images_real_files]
    images_fake = [Image.open(file) for file in images_fake_files]

    real = np.stack(images_real)
    fake = np.stack(images_fake)

    print('Prepared', real.shape, fake.shape)
    # convert integer to floating point values
    real = real.astype('float32')
    fake = fake.astype('float32')
    # resize images
    inception_real = scale_images(real, (256, 256, 3))
    inception_fake = scale_images(fake, (256, 256, 3))
    # pre-process images
    inception_real = preprocess_input(inception_real)
    inception_fake = preprocess_input(inception_fake)
    # fid between real and fake
    fid = calculate_frechet_distance(inception, inception_real, inception_fake)
    print('FID: %.3f' % fid)
    fake_inception_score = calculate_inception_score(inception, fake)
    print('Inception: %.3f' % fake_inception_score)
    # fad between real and fake
    vgg_real = scale_images(real, (224, 224, 3))
    vgg_fake = scale_images(fake, (224, 224, 3))
    vgg_real = tensorflow.keras.applications.vgg19.preprocess_input(vgg_real)
    vgg_fake = tensorflow.keras.applications.vgg19.preprocess_input(vgg_fake)
    fad = calculate_frechet_distance(vgg, vgg_real, vgg_fake)
    print('FAD: %.3f' % fad)

    return fid, fad, fake_inception_score
