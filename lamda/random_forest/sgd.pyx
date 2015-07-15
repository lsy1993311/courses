# __author__ = 'guoxy'
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
import numpy.random as random

@cython.boundscheck(False)
def sgd(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.long_t, ndim=1] Y, double ep):
    # randomly divide positive and negative classes
    cdef long pc = Y[random.randint(0, len(Y))]
    cdef np.ndarray pos_idx = np.where(Y == pc)[0]
    cdef np.ndarray neg_idx = np.where(Y != pc)[0]
    if len(neg_idx) == 0:
        return None, None, None

    cdef np.ndarray xp = X[pos_idx[0]]
    cdef np.ndarray xn = X[neg_idx[0]]

    cdef np.ndarray w = xp - xn
    w /= np.linalg.norm(w)
    cdef double b = 1
    cdef int T = max(np.int(1 / ep ** 2), 1)
    cdef list idx = list(xrange(len(X)))
    # random.shuffle(idx)

    # SGD
    cdef np.ndarray w_previous = w.copy() + 1
    cdef int t = 0
    while t < T:
        if np.linalg.norm(w_previous - w) <= 0.0001:
            break
        else:
            w_previous[:] = w
            random.shuffle(idx)

        for j in idx:
            lx = 2 * (Y[j] == pc) - 1
            if (w.dot(X[j].T) + b) * lx < 0:
                step_size = 0.1 / (0.1 + t * ep ** 2)
                w += step_size * X[j] * lx
                b += step_size * lx
            t += 1

    return w, b, pc