import tensorflow as tf
from tensorflow.keras.models import Model
from network.conv import *
from network.dataset_util import *


def GeneratorFn(
        input,                  # Has size [batch_size, samples, features]
        z,                      # Noise vector [batch_size, 100]
        samples,
        feature_size,
        z_dim,
        r_scale,
        conv_size=32,
        filters=128):


    # First, take the input and compress it down into a [batch_size, 100] vector, equivalent size to noise vec.

    conv1 = temporal_conv_downsample(input, filters, conv_size, r_scale)
    samples //= r_scale

    conv2 = temporal_conv_downsample(conv1, filters, conv_size, r_scale)
    samples //= r_scale

    conv3 = temporal_conv_downsample(conv2, filters, conv_size, r_scale)
    samples //= r_scale

    conv3_flat = tf.keras.layers.Flatten()(conv3)
    dense = tf.keras.layers.Dense(z_dim, activation='relu', name="Dense1")(conv3_flat)
    encoded = tf.keras.layers.BatchNormalization()(dense)


    # Concatenate the noise and do a fully connected network to begin decompressing
    encoded = tf.keras.layers.concatenate([encoded, z], axis=1)
    unencoded = tf.keras.layers.Dense(samples*filters, activation='relu')(encoded)
    unencoded_norm = tf.keras.layers.BatchNormalization()(unencoded)

    unencoded_norm_rs = tf.reshape(unencoded_norm, [-1, samples, filters], name="ReshapeUnencoded")
    conv3_id = tf.keras.layers.Lambda(lambda x : x, name="DownConv3_skip")(conv3)
    skip0 = tf.concat([unencoded_norm_rs, conv3_id], axis=2)

    # Now do temporal deconvolutions

    deconv1 = temporal_deconv(skip0, filters, r_scale, name="UpConv1")
    skip1 = tf.keras.layers.concatenate([deconv1, conv2], axis=2, name="Concat1")
    samples *= r_scale

    deconv2 = temporal_deconv(skip1, filters, r_scale, name="UpConv2")
    skip2 = tf.keras.layers.concatenate([deconv2, conv1], axis=2, name="Concat2")
    samples *= r_scale
    # Collapse down to a single audio channel.
    deconv3 = temporal_deconv(skip2, 1, r_scale, name="UpConv3")
    samples *= r_scale
    output = tf.reshape(deconv3, [-1, samples], name="Signal")
    output_plus_features = MelSpecFeatures(feature_size)(output)
    return output_plus_features

def GeneratorModel(sample_size, feature_size, z_dim, r_scale, filter_dim):
    in1 = tf.keras.layers.Input(shape=(sample_size, feature_size + 1), name="Cond_in")
    in2 = tf.keras.layers.Input(shape=(z_dim), name="Z_in")
    value = GeneratorFn(in1, in2, sample_size, feature_size, z_dim, r_scale, filter_dim)
    model = Model(inputs=[in1, in2], outputs=value)
    return model