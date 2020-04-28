import network.generator as generator
import tensorflow as tf
import numpy as np
import os

class GeneratorTest(tf.test.TestCase):
    def setUp(self):
        super(GeneratorTest, self).setUp()

    def testGeneratorOutput(self):
        model = generator.GeneratorModel(sample_size=4096, feature_size=32, z_dim=100, r_scale=8, filter_dim=64)
        fake_input = tf.random.uniform((10, 4096, 33))
        fake_rand = tf.random.uniform((10, 100))
        result = model.predict({"Cond_in": fake_input, "Z_in": fake_rand})
        self.assertAllEqual(tf.shape(result), [10, 4096, 33])
        self.assertNotEqual(tf.math.count_nonzero(result), 0)


if __name__ == '__main__':
    tf.test.main()