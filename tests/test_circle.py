import numpy as np
import unittest
from utils.circle import Circle


class TestCircle(unittest.TestCase):
    circle_1 = Circle(0, 0, 1)
    circle_2 = Circle(0, 0, 5)

    def test_numpy_conversion(self):
        self.assertTrue(
            np.all(TestCircle.circle_1.to_numpy() == np.array([0, 0, 1])))

    def test_is_within(self):
        self.assertTrue(TestCircle.circle_1.is_within(TestCircle.circle_2))
        self.assertFalse(TestCircle.circle_2.is_within(TestCircle.circle_1))


if __name__ == '__main__':
    unittest.main()
