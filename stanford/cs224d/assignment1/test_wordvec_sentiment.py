__author__ = 'guoxy'

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from wordvec_sentiment import *

class MyTestCase(unittest.TestCase):

    def test_softmax(self):
        assert_array_almost_equal(softmax(np.array([[1001, 1002], [3, 4]])),
                                  np.array([[0.26894142, 0.73105858],
                                            [0.26894142, 0.73105858]]))
        assert_array_almost_equal(softmax(np.array([[-1001, -1002]])),
                                  [[0.73105858, 0.26894142]])
        assert_array_almost_equal(softmax(np.array([1, 2])),
                                  np.array([0.26894142, 0.73105858]))

    def test_sigmoid_and_grad(self):
        x = np.array([[1, 2], [-1, -2]])
        f = sigmoid(x)
        g = sigmoid_grad(f)
        assert_array_almost_equal(f,
                                  np.array([[0.731059, 0.880797],
                                            [0.268941, 0.119203]]))
        assert_array_almost_equal(g, np.array([[0.196612, 0.104994],
                                               [0.196612, 0.104994]]))

    def test_gradcheck_naive(self):
        # Sanity check for the gradient checker
        quad = lambda x: (np.sum(x ** 2), x * 2)

        print "=== For autograder ==="
        gradcheck_naive(quad, np.array(123.456))      # scalar test
        gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
        gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test



if __name__ == '__main__':
    unittest.main()
