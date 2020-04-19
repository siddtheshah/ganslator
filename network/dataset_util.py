import tensorflow as tf
import tensorflow_io as tfio
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""
Creates a dataset using .wav or .mp3 files in a given directory.
Input:
    a path to an audio file directory
Output:
    a tf.data.Dataset of audio tensors.
"""


def create_dataset_from_io_spec(input_spec, output_spec, batch_size=16):
    input_fs = tf.data.Dataset.list_files(input_spec)
    output_fs = tf.data.Dataset.list_files(output_spec)
    input_audio = input_fs.shuffle(buffer_size=1024).map(load_audio, num_parallel_calls=AUTOTUNE)
    input_batch = input_audio.repeat().batch(batch_size)
    output_batch = output_fs.shuffle(buffer_size=1024).map(load_audio, num_parallel_calls=AUTOTUNE).repeat().batch(
        batch_size)
    feature_batch = input_audio.map(calculate_features, num_parallel_calls=AUTOTUNE).repeat().batch(batch_size)

    return tf.data.Dataset.zip((input_batch, feature_batch, output_batch))

def load_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16384)
    audio = tf.squeeze(audio)
    return audio

def calculate_features(audio):
    batch_size, num_samples, sample_rate = 1, 16384, 16000.0

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(audio, frame_length=128, frame_step=32)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 32
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)

    mel_spectrograms = tf.matmul(
        spectrograms, linear_to_mel_weight_matrix)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    return mel_spectrograms
