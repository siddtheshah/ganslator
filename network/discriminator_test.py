import network.discriminator as discriminator
import tensorflow as tf
import numpy as np
import os

class DiscriminatorTest(tf.test.TestCase):
    def setUp(self):
        super(DiscriminatorTest, self).setUp()

    def testDiscriminatorOutput(self):
        model = discriminator.DiscriminatorModel(sample_size=4096, feature_size=32, r_scale=8, filter_dim=64)
        fake_input = tf.random.uniform((10, 4096, 33))
        result = model.predict({"input": fake_input})
        print(result)
        self.assertNotEqual(tf.math.count_nonzero(result), 0)


if __name__ == '__main__':
    tf.test.main()