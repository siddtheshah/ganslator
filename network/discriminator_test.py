import network.discriminator as discriminator
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

class DatasetTest(tf.test.TestCase):
    def setUp(self):
        super(DatasetTest, self).setUp()

    def testDatasetOutput(self):
        path = Path(__file__).parent
        ds = ds_util.create_dataset_from_audio_dir(os.path.join(path, "testdata*"), 1)

        iter = ds.__iter__()
        first_item = iter.get_next()

        expected_shape = np.array([1, 16384, 1])

        # Make sure the dataset is reading data correctly.
        self.assertAllEqual(expected_shape, tf.shape(first_item.audio))
        self.assertNotEqual(0, first_item.audio[0][0][0])



if __name__ == '__main__':
    tf.test.main()