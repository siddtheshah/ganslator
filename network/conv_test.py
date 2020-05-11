import network.conv as nc
import tensorflow as tf
import numpy as np
import os

class ConvTest(tf.test.TestCase):
    def setUp(self):
        super(DatasetTest, self).setUp()

    def testMelSpecLayer(self):
        example_input = tf.random.uniform((10, 16384))

        input_layer = tf.keras.layers.Input(shape=[16384])
        out = nc.MelSpecFeatures(32)(input_layer)

        mel_model = tf.keras.Model(input_layer, out)

        features = mel_model.predict(example_input)

        expected_feature_shape = np.array([10, 16384, 33])
        self.assertAllEqual(expected_feature_shape, tf.shape(features))
        self.assertNotEqual(0, tf.math.count_nonzero(features))

    def testTemporalConvDown(self):
        x = tf.keras.layers.Input((4096, 33))
        y = nc.temporal_conv_downsample(x, num_filters=32, kernel_length=16, reduce_factor=32)
        model = tf.keras.Model(x, y)


        fake_input = tf.random.uniform((10, 4096, 33))
        output = model.predict(fake_input)
        expected_shape = np.array([10, 128, 32])
        self.assertAllEqual(expected_shape, tf.shape(output))

    def testTemporalConvUp(self):
        x = tf.keras.layers.Input((128, 32))
        y = nc.temporal_deconv(x, num_filters=32, enlarge_factor=32)
        model = tf.keras.Model(x, y)


        fake_input = tf.random.uniform((10, 128, 32))
        output = model.predict(fake_input)
        expected_shape = np.array([10, 4096, 32])
        self.assertAllEqual(expected_shape, tf.shape(output))

if __name__ == '__main__':
    tf.test.main()
