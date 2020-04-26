import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import tensorflow_addons as tfa

# If the input is (batch_size, samples, features), then the output will be (batch_size, reduced_samples, filters)
# And then instance norm, of course.

# ONLY CONVOLVES OVER TIME DIMENSION
def temporal_conv_downsample(layer_input, num_filters, kernel_length, reduce_factor, name=None):
    d = tf.keras.layers.Conv1D(num_filters, kernel_length, padding='same', name=name)(layer_input)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    d = tfa.layers.InstanceNormalization(axis=2,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(d)


    d = tf.keras.layers.Conv1D(num_filters, reduce_factor, strides=reduce_factor, padding='valid', name=name)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    d = tfa.layers.InstanceNormalization(axis=2,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(d)
    return d

def temporal_deconv(layer_input, num_filters, enlarge_factor, name=None):
    d = tf.keras.layers.UpSampling1D(size=enlarge_factor)(layer_input)
    # Use adjacent values pre-upsample to interpolate.
    d = tf.keras.layers.Conv1D(num_filters, 3*enlarge_factor, padding='same', name=name)(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    d = tfa.layers.InstanceNormalization(axis=2,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(d)

    return d
