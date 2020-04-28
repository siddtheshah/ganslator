import network.dataset_util as ds_util
import network.combined_model as combined_model
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

class CombinedModelTest(tf.test.TestCase):
    def setUp(self):
        super(CombinedModelTest, self).setUp()

    def testCombinedModelTest(self):
        path = Path(__file__).parent
        ganslator = combined_model.GANslator()
        dataset = ds_util.create_dataset_from_io_spec(os.path.join(path, "more_testdata/*"), os.path.join(path, "more_testdata/*"))
        ganslator.train(dataset, 2)


        # Make sure the dataset is reading data correctly.
        # self.assertAllEqual(expected_input_shape, tf.shape(input))
        # self.assertAllEqual(expected_input_shape, tf.shape(output))


if __name__ == '__main__':
    tf.test.main()