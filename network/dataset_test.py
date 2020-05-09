import network.dataset_util as ds_util
import tensorflow as tf
import numpy as np
import os
from pathlib import Path


class DatasetTest(tf.test.TestCase):
    def setUp(self):
        super(DatasetTest, self).setUp()

    def testDatasetOutput(self):
        path = str(Path(__file__).parent)
        ds = ds_util.create_unconditioned_dataset_from_io_spec(os.path.join(path, "testing/testdata*"),
                                                               os.path.join(path, "testing/testdata*"), 16384).batch(1)

        iter = ds.__iter__()
        example = iter.get_next()
        input = example[0]
        output = example[1]

        expected_input_shape = np.array([1, 16384])

        # Make sure the dataset is reading data correctly.
        self.assertAllEqual(expected_input_shape, tf.shape(input))
        self.assertAllEqual(expected_input_shape, tf.shape(output))

    def testChunkedAudio(self):
        path = str(Path(__file__).parent)
        fn = ds_util.audio_chunking_fn(chunk_size=1024, misalignment=200, starting_offset=1000)
        audio_chunks = fn(os.path.join(path, "testing", "more_testdata", "Clap_00006.wav"))
        first = audio_chunks[0]
        second = audio_chunks[1]

        expected_input_shape = np.array([1024])
        # Make sure the dataset is reading data correctly.
        self.assertAllEqual(expected_input_shape, tf.shape(first))
        self.assertAllEqual(expected_input_shape, tf.shape(second))


if __name__ == '__main__':
    tf.test.main()
