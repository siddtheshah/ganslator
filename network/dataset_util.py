import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def is_matching_ravdess_emotion_file(file_path, emotion):
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    num_strs = name.split('-')
    nums = [int(x) for x in num_strs]
    if nums[1] == 2:  # Is singing
        return False
    if nums[3] == 2:  # Is high intensity
        return False
    if nums[2] != emotion:
        return False
    return True

"""
Creates a Dataset from the standard ravdess directory structure. 
Input:
    a path to the top level ravdess directory
"""

def create_dataset_from_ravdess(ravdess_dir, emotion_input=0, emotion_output=3):
    files = [f for f in os.listdir(ravdess_dir) if os.path.isfile(os.path.join(ravdess_dir, f))]
    input_fs = [x for x in files if is_matching_ravdess_emotion_file(x, emotion_input)]
    output_fs = [x for x in files if is_matching_ravdess_emotion_file(x, emotion_output)]
    input_ds = tf.data.Dataset.from_tensor_slices(input_fs)
    output_ds = tf.data.Dataset.from_tensor_slices(output_fs)
    input_wav = input_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    output_wav = output_ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    return tf.data.Dataset.zip((input_wav, output_wav))

"""
Creates a dataset using .wav or .mp3 files in a given directory. Remember to call .batch() to batch this dataset.
Input:
    a path to an audio file directory
Output:
    a tf.data.Dataset of audio tensors.
"""

def create_unconditioned_dataset_from_io_spec(input_spec, output_spec):
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