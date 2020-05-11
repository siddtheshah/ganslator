import eval.inception
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3

class EvalTest(tf.test.TestCase):
    def setUp(self):
        super(EvalTest, self).setUp()

    def testInceptionComputation(self):
        input = tf.random.uniform((5, 256, 256, 3))

        model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))

        inception_scores = eval.inception.calculate_inception_score(model, input)
        print(inception_scores)

    def testFIDComputation(self):
        input = tf.random.uniform((5, 256, 256, 3))

        model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))

        self_fid = eval.inception.calculate_frechet_distance(model, input, input)
        print(self_fid)

if __name__ == '__main__':
    tf.test.main()
