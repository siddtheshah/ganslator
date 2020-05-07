import tensorflow as tf
from network.conv import *

"""
  Input: [batch_size, samples, features] representing the real or generated audio. 
  Output: [0-1] indicating whether the output is part of domain A or B.
"""


def DiscriminatorFn(
        input,  # Has size [batch_size, samples, features]
        samples,
        r_scale,
        conv_size=32,
        filters=64):
    # Take the input and convolve over the time dimension several times. Collapse filters and get a score of 0-1 at the end.
    # TODO: Look into phase shuffling

    conv1 = temporal_conv_downsample(input, filters, conv_size, r_scale)
    samples //= r_scale

    conv2 = temporal_conv_downsample(conv1, filters, conv_size, r_scale)
    samples //= r_scale

    conv3 = temporal_conv_downsample(conv2, filters, conv_size, r_scale)
    samples //= r_scale

    conv3_flat = tf.keras.layers.Flatten()(conv3)

    dense_intermediate = tf.keras.layers.Dense(filters)(conv3_flat)
    output = tf.keras.layers.Dense(1)(dense_intermediate)
    return output


def DiscriminatorModel(sample_size, feature_size, r_scale, filter_dim):
    input = tf.keras.layers.Input(shape=(sample_size, feature_size + 1), name="input")
    value = DiscriminatorFn(input, sample_size, r_scale, filter_dim)
    return Model(inputs=input, outputs=value)
