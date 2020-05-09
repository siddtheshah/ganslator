import tensorflow as tf
import os
import glob

AUTOTUNE = tf.data.experimental.AUTOTUNE

def collect_matching_ravdess_files(ravdess_dir, emotion_input, emotion_output):
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

    # print(ravdess_dir)
    glob_pattern = os.path.join(ravdess_dir, "**")
    # print(glob_pattern)
    globbed = glob.glob(glob_pattern, recursive=True)
    # print(globbed)
    files = [f for f in globbed  if os.path.isfile(f)]
    # files = ["fake.wav", "faker.wav"]
    input_fs = [x for x in files if is_matching_ravdess_emotion_file(x, emotion_input)]
    output_fs = [x for x in files if is_matching_ravdess_emotion_file(x, emotion_output)]
    # input_fs = ["network/testing/testdata.wav"]
    # output_fs = ["network/testing/testdata.wav"]
    # print(input_fs)
    # print(output_fs)
    input_ds = tf.data.Dataset.from_tensor_slices(input_fs)
    output_ds = tf.data.Dataset.from_tensor_slices(output_fs)
    return input_ds, output_ds

"""
Creates a Dataset from the standard ravdess directory structure. 
Input:
    a path to the top level ravdess directory
"""

def basic_ravdess(ravdess_dir, samples, emotion_input=1, emotion_output=3):
    input_ds, output_ds = collect_matching_ravdess_files(ravdess_dir, emotion_input, emotion_output)
    input_wav = input_ds.map(load_audio_fn(samples), num_parallel_calls=AUTOTUNE)
    output_wav = output_ds.map(load_audio_fn(samples), num_parallel_calls=AUTOTUNE)
    return tf.data.Dataset.zip((input_wav, output_wav))

def chunked_ravdess(ravdess_dir, chunk_size, misalignment, starting_offset, emotion_input=1, emotion_output=3):
    input_ds, output_ds = collect_matching_ravdess_files(ravdess_dir, emotion_input, emotion_output)
    input_wav = input_ds.flat_map(audio_chunking_fn(chunk_size, misalignment, starting_offset))
    output_wav = output_ds.flat_map(audio_chunking_fn(chunk_size, misalignment, starting_offset))
    return tf.data.Dataset.zip((input_wav, output_wav))

"""
Creates a dataset using .wav or .mp3 files in a given directory. Remember to call .batch() to batch this dataset.
Input:
    a path to an audio file directory
Output:
    a tf.data.Dataset of audio tensors.
"""

def create_unconditioned_dataset_from_io_spec(input_spec, output_spec, samples):
    input_fs = tf.data.Dataset.list_files(input_spec)
    output_fs = tf.data.Dataset.list_files(output_spec)
    input_batch = input_fs.shuffle(buffer_size=1024).map(load_audio_fn(samples), num_parallel_calls=AUTOTUNE)
    output_batch = output_fs.shuffle(buffer_size=1024).map(load_audio_fn(samples), num_parallel_calls=AUTOTUNE)
    # feature_batch = input_audio.map(calculate_mel_spec_features, num_parallel_calls=AUTOTUNE).repeat()

    return tf.data.Dataset.zip((input_batch, output_batch))

def load_audio_fn(samples):
    return lambda x: load_audio(x, samples)

def load_audio(file_path, samples):
    # tf.print(file_path)
    audio = tf.io.read_file(file_path)
    # tf.print(tf.shape(audio))
    audio, sr = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=samples)
    audio = tf.squeeze(audio)
    return audio

# Misalignment should be specified as a layer, but that would require reading in an unchunked tensor
# into the model itself. We instead do it at the DS level.
def audio_chunking_fn(chunk_size, misalignment, starting_offset):
    return lambda x: load_chunked_audio(x, chunk_size, misalignment, starting_offset)

def load_chunked_audio(file_path, chunk_size, misalignment, starting_offset):
    audio = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=-1)
    audio = tf.squeeze(audio)
    audio_size = tf.shape(audio)[0]
    num_samples = audio_size - starting_offset - misalignment*10
    num_chunks = tf.cast(num_samples / chunk_size, tf.int32)
    r = tf.random.normal([1])
    shift = tf.cast(misalignment * r, tf.int32)
    start = starting_offset + shift
    size = tf.convert_to_tensor(num_chunks * chunk_size)
    audio_slice = tf.slice(audio, start, [size])
    rs = tf.reshape(audio_slice, [-1, chunk_size])
    return tf.unstack(rs)

