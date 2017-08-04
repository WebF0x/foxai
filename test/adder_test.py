from foxai import adder_ai
import unittest


class AdderTest(unittest.TestCase):
    def test_adder_learned_to_add(self):
        adder = adder_ai.AdderAI()
        adder.train()

        self.assertAlmostEquals(1, adder.add(1, 0), delta=0.1)
        self.assertAlmostEquals(1, adder.add(0, 1), delta=0.1)
        self.assertAlmostEquals(0, adder.add(0, 0), delta=0.1)
        self.assertAlmostEquals(5, adder.add(2, 3), delta=0.1)
        self.assertAlmostEquals(-5, adder.add(-2, -3), delta=0.1)

    def test_adder_can_train_twice_without_breaking(self):
        adder = adder_ai.AdderAI()
        adder.train()
        adder.train()
