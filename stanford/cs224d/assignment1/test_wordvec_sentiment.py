__author__ = 'guoxy'
import unittest
import numpy as np
import random

from numpy.testing import assert_array_almost_equal
from wordvec_sentiment import *

from cs224d.data_utils import *
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### YOUR CODE HERE: try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it
        ### possible to test cost functions with built in randomness later

        x[ix] += h
        random.setstate(rndstate)
        fx_p, grad_p = f(x)
        x[ix] -= 2 * h
        random.setstate(rndstate)
        fx_n, grad_n = f(x)

        numgrad = (fx_p - fx_n) / (h * 2)
        x[ix] += h

        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return False

        it.iternext() # Step to next dimension

    print "Gradient check passed!"
    return True


class Word2VecTestCase(unittest.TestCase):

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

    def test_forward_backward_prop(self):
        N = 20
        dimensions = [10, 5, 10]
        data = np.random.randn(N, dimensions[0])   # each row will be a datum
        labels = np.zeros((N, dimensions[2]))
        for i in xrange(N):
            labels[i, random.randint(0, dimensions[2]-1)] = 1

        params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )

        print "=== Testing forward_backward_prop ==="
        assert gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)



if __name__ == '__main__':
    unittest.main()
