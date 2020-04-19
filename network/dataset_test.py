import network.dataset_util as ds_util
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

class DatasetTest(tf.test.TestCase):
    def setUp(self):
        super(DatasetTest, self).setUp()

    def testDatasetOutput(self):
        path = Path(__file__).parent
        ds = ds_util.create_dataset_from_io_spec(os.path.join(path, "testdata*"), os.path.join(path, "testdata*"), 1)

        iter = ds.__iter__()
        example = iter.get_next()
        input = example[0]
        features = example[1]

        expected_input_shape = np.array([1, 16384])
        expected_feature_shape = np.array([1, 16384, 32])

        # Make sure the dataset is reading data correctly.
        self.assertAllEqual(expected_input_shape, tf.shape(input))
        self.assertNotEqual(0, tf.math.count_nonzero(features))



if __name__ == '__main__':
    tf.test.main()