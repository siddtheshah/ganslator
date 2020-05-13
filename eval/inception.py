import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
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
        # resize
        new_image = resize(image, new_shape)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def calculate_inception_score(model, images):
    preds = model.predict(images)
    kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
    kl = np.mean(np.sum(kl, 1))

    return np.exp(kl)

# calculate frechet inception distance
def calculate_frechet_distance(model, images1, images2):
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

def score(model_results_dir, max_images=200):
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))
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
    real = scale_images(real, (256, 256, 3))
    fake = scale_images(fake, (256, 256, 3))
    print('Scaled', real.shape, fake.shape)
    # pre-process images
    real = preprocess_input(real)
    fake = preprocess_input(fake)
    # fid between real and real
    self_fid = calculate_frechet_distance(model, real, real)
    print('FID (same): %.3f' % self_fid)
    # fid between real and fake
    cross_fid = calculate_frechet_distance(model, real, fake)
    print('FID (different): %.3f' % cross_fid)
    fake_inception_score = calculate_inception_score(model, fake)
    print('Inception: %.3f' % fake_inception_score)

    return self_fid, cross_fid, fake_inception_score
