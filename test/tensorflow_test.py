import tensorflow as tf


class TensorFlowTest(tf.test.TestCase):
    def test_square(self):
        with self.test_session():
            x = tf.square([2, 3])
            y = x.eval()
            self.assertAllEqual([4, 9], y)

