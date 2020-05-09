import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import tensorflow_addons as tfa

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, depth, heads):
        self.heads = heads
        self.depth = depth


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

class MelSpecFeatures(tf.keras.layers.Layer):
    def __init__(self, num_mel_bins):
        self.num_mel_bins = num_mel_bins
        super(MelSpecFeatures, self).__init__()

    def call(self, x):
        sample_rate = 16000.0

        # A 1024-point STFT with frames of 64 ms and 75% overlap.
        stfts = tf.signal.stft(x, frame_length=128, frame_step=1, pad_end=True)

        # Don't try to propagate gradients through the stft.
        spectrograms = tf.stop_gradient(tf.abs(stfts))

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
            upper_edge_hertz)

        mel_spectrograms = tf.matmul(
            spectrograms, linear_to_mel_weight_matrix)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
        exp = tf.expand_dims(x, 2)
        output = tf.keras.layers.concatenate([exp, mel_spectrograms], axis=-1)
        # print(output.get_shape())
        return output

    def from_config(self, config):
        return MelSpecFeatures(config['num_mel_bins'])

    def get_config(self):
        return {'num_mel_bins': self.num_mel_bins}

    def compute_output_shape(self, input_shape):
        print(input_shape[1])
        print(self.num_mel_bins)
        output = tf.convert_to_tensor([-1, input_shape[1], 1 + self.num_mel_bins])
        # tf.print(output)
        # print(output.get_shape())
        return output

