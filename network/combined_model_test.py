import network.dataset_util as ds_util
import network.combined_model as combined_model
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

class CombinedModelTest(tf.test.TestCase):
    def setUp(self):
        super(CombinedModelTest, self).setUp()

    def testModelTrainableSaveable(self):

        path = str(Path(__file__).parent)
        ganslator = combined_model.GANslator(sample_size=8192)
        dataset = ds_util.create_unconditioned_dataset_from_io_spec(os.path.join(path, "testing/more_testdata/*"), os.path.join(path, "testing/more_testdata/*"), samples=8192)
        ganslator.train(dataset, 2, save_interval=1)
        model_save_path = os.path.join(path, "testing")
        ganslator.save_to_path(model_save_path)
        ganslator.load_from_path(model_save_path)
        ganslator.train(dataset, 2, save_interval=1)



if __name__ == '__main__':
    tf.test.main()
