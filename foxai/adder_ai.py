import tensorflow as tf


class AdderAI:
    def __init__(self):
        self.weight1 = None
        self.weight2 = None
        self.biaslol = None

        # TensorFlow graph
        self.x1 = tf.placeholder(tf.float32)
        self.x2 = tf.placeholder(tf.float32)
        self.w1 = tf.Variable([0.0], tf.float32)
        self.w2 = tf.Variable([0.0], tf.float32)
        self.bias = tf.Variable([10.0], tf.float32)
        self.model = self.x1 * self.w1 + self.x2 * self.w2 + self.bias
        self.y = tf.placeholder(tf.float32)

        error = tf.abs(self.model - self.y)
        relative_error = error / self.y
        self.loss = tf.reduce_sum(tf.square(relative_error))

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train_operation = optimizer.minimize(self.loss)

        self.init_operation = tf.global_variables_initializer()

    def train(self):
        training_data = self.get_training_data()
        with tf.Session() as session:
            session.run(self.init_operation)
            while True:
                session.run(self.train_operation, training_data)
                return_parameters = [self.w1, self.w2, self.bias, self.loss]
                self.weight1, self.weight2, self.biaslol, curr_loss = session.run(return_parameters, training_data)
                if curr_loss < 0.0001:
                    break

    def add(self, x1, x2):
        model_parameters = {self.x1: x1,
                            self.x2: x2,
                            self.w1: self.weight1,
                            self.w2: self.weight2,
                            self.bias: self.biaslol}
        with tf.Session() as session:
            output = session.run(self.model, model_parameters)
        return output

    def get_training_data(self):
        training_cases = [
            [1, 2, 3],
            [-1, -2, -3],
            [0, 2, 2],
            [100, 200, 300],
            [-100, -200, -300],
            [0, 200, 200]
        ]

        x1 = [training_case[0] for training_case in training_cases]
        x2 = [training_case[1] for training_case in training_cases]
        y = [training_case[2] for training_case in training_cases]

        return {self.x1: x1,
                self.x2: x2,
                self.y: y}
