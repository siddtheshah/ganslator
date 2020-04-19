import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import tensorflow_addons as tfa

"""
  Input: [None, audio_extracted_features]
  Output: [None, slice_len, 1]
"""

def GeneratorFn(
        input,                  # Has size [batch_size, samples, features]
        z,                      # Noise vector [batch_size, 100]
        samples,
        z_dim,
        r_scale,
        filters=128):
    batch_size = tf.shape(input)[0]
    conv_size = 32


    # First, take the input and compress it down into a [batch_size, 100] vector, equivalent size to noise vec.
    # To do this, we convolve over time, then do a convolution with the kernel height covering the entire spectrogram.

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
    def temporal_deconv(layer_input, num_filters, enlarge_factor, name=None):
        d = tf.keras.layers.UpSampling1D(size=enlarge_factor)(layer_input)
        d = tf.keras.layers.Conv1D(num_filters, 3*enlarge_factor, padding='same', name=name)(d)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        d = tfa.layers.InstanceNormalization(axis=2,
                                             center=True,
                                             scale=True,
                                             beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(d)

        return d

    deconv1 = temporal_deconv(skip0, filters, r_scale, name="UpConv1")
    skip1 = tf.keras.layers.concatenate([deconv1, conv2], axis=2, name="Concat1")
    samples *= r_scale

    deconv2 = temporal_deconv(skip1, filters, r_scale, name="UpConv2")
    skip2 = tf.keras.layers.concatenate([deconv2, conv1], axis=2, name="Concat2")
    samples *= r_scale

    deconv3 = temporal_deconv(skip2, 1, r_scale, name="UpConv3")
    samples *= r_scale
    output = tf.squeeze(deconv3)
    return output

def GeneratorModel(sample_size, feature_size, z_dim, r_scale, filter_dim):
    in1 = tf.keras.layers.Input(shape=(sample_size, feature_size), name="Cond_in")
    in2 = tf.keras.layers.Input(shape=(z_dim), name="Z_in")
    value = GeneratorFn(in1, in2, sample_size, z_dim, r_scale, filter_dim)
    model = Model(inputs=[in1, in2], outputs=value)
    return model


# def build_generator(self):
#     """U-Net Generator"""
#
#     def conv2d(layer_input, filters, f_size=4):
#         """Layers used during downsampling"""
#         d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
#         d = LeakyReLU(alpha=0.2)(d)
#         d = InstanceNormalization()(d)
#         return d
#
#     def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
#         """Layers used during upsampling"""
#         u = UpSampling2D(size=2)(layer_input)
#         u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
#         if dropout_rate:
#             u = Dropout(dropout_rate)(u)
#         u = InstanceNormalization()(u)
#         u = Concatenate()([u, skip_input])
#         return u
#
#     # Image input
#     d0 = Input(shape=self.img_shape)
#
#     # Downsampling
#     d1 = conv2d(d0, self.gf)
#     d2 = conv2d(d1, self.gf * 2)
#     d3 = conv2d(d2, self.gf * 4)
#     d4 = conv2d(d3, self.gf * 8)
#
#     # Upsampling
#     u1 = deconv2d(d4, d3, self.gf * 4)
#     u2 = deconv2d(u1, d2, self.gf * 2)
#     u3 = deconv2d(u2, d1, self.gf)
#
#     u4 = UpSampling2D(size=2)(u3)
#     output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
#
#     return Model(d0, output_img)
