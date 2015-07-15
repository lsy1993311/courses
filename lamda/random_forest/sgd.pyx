# __author__ = 'guoxy'
# Cythonized SGD for the training of tree nodes
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
import numpy.random as random

XTYPE = np.double
ctypedef np.double_t XTYPE_t
YTYPE = np.int64
ctypedef np.int64_t YTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
def sgd(np.ndarray[XTYPE_t, ndim=2] X not None,
        np.ndarray[YTYPE_t, ndim=1] Y not None,
        double ep):

    assert X.dtype == XTYPE and Y.dtype == YTYPE

    # randomly divide positive and negative classes
    cdef long pc = Y[random.randint(0, len(Y))]
    cdef np.ndarray pos_idx = np.where(Y == pc)[0]
    cdef np.ndarray neg_idx = np.where(Y != pc)[0]
    if len(neg_idx) == 0:
        return None, None, None

    # variables initialization
    cdef np.ndarray[XTYPE_t, ndim=1] xp = X[pos_idx[0]]
    cdef np.ndarray[XTYPE_t, ndim=1] xn = X[neg_idx[0]]

    cdef np.ndarray[XTYPE_t, ndim=1] w = xp - xn
    w /= np.linalg.norm(w)
    cdef double b = (w.dot(xp) + w.dot(xn)) / 2
    cdef unsigned int T = max(np.int(1 / ep ** 2), 1)
    cdef np.ndarray[np.int64_t, ndim=1] idx = np.array(xrange(len(X)))

    # SGD
    cdef np.ndarray[XTYPE_t, ndim=1] w_previous = w.copy() + 1
    cdef unsigned int t = 0
    cdef np.int64_t j = 0
    cdef double step_size = 0
    cdef YTYPE_t lx = 0

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