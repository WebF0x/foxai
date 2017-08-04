from foxai import adder_ai
import unittest


class AdderTest(unittest.TestCase):
    def setUp(self):
        self.adder = adder_ai.AdderAI()

    def test_adder_learns_to_add(self):
        self.adder.train()

        self.assertAlmostEquals(1, self.adder.add(1, 0), delta=0.1)
        self.assertAlmostEquals(1, self.adder.add(0, 1), delta=0.1)
        self.assertAlmostEquals(0, self.adder.add(0, 0), delta=0.1)
        self.assertAlmostEquals(5, self.adder.add(2, 3), delta=0.1)
        self.assertAlmostEquals(-5, self.adder.add(-2, -3), delta=0.1)

    def test_adder_can_train_twice(self):
        self.adder.train()
        self.adder.train()
