import tensorflow as tf
import tensorflow_io as tfio
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""
Creates a dataset using .wav or .mp3 files in a given directory. Remember to call .batch() to batch this dataset.
Input:
    a path to an audio file directory
Output:
    a tf.data.Dataset of audio tensors.
"""

def create_dataset_from_io_spec(input_spec, output_spec):
    input_fs = tf.data.Dataset.list_files(input_spec)
    output_fs = tf.data.Dataset.list_files(output_spec)
    input_audio = input_fs.shuffle(buffer_size=1024).map(load_audio, num_parallel_calls=AUTOTUNE)
    input_batch = input_audio.repeat()
    output_batch = output_fs.shuffle(buffer_size=1024).map(load_audio, num_parallel_calls=AUTOTUNE).repeat()
    # feature_batch = input_audio.map(calculate_mel_spec_features, num_parallel_calls=AUTOTUNE).repeat()

    return tf.data.Dataset.zip((input_batch, output_batch))

def load_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16384)
    audio = tf.squeeze(audio)
    return audio

class MelSpecFeatures(tf.keras.layers.Layer):
    def __init__(self, num_mel_bins):
        self.num_mel_bins = num_mel_bins
        super(MelSpecFeatures, self).__init__()

    def call(self, x):
        batch_size, num_samples, sample_rate = 1, 16384, 16000.0

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
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1 + self.num_mel_bins)
