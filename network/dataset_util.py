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
    input_batch = input_fs.shuffle(buffer_size=1024).map(load_audio, num_parallel_calls=AUTOTUNE)
    output_batch = output_fs.shuffle(buffer_size=1024).map(load_audio, num_parallel_calls=AUTOTUNE)
    # feature_batch = input_audio.map(calculate_mel_spec_features, num_parallel_calls=AUTOTUNE).repeat()

    return tf.data.Dataset.zip((input_batch, output_batch))

def load_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16384)
    audio = tf.squeeze(audio)
    return audio