#__author__ = 'guoxy'
from __future__ import division
from collections import deque, Counter
import numpy as np
import numpy.random as random
# __all__ = ["RandomForest"]


class TreeNode(object):

    def __init__(self, err, depth):
        self.depth = depth

        self.err = err
        self.weight = None
        self.bias = None
        self.pos_child = None
        self.neg_child = None
        self.class_distr = None
        self.positive_class = self.negative_class = None
        self.is_terminal = False

    @property
    def classifier(self):
        return self.weight, self.bias

    def direct(self, prediction):
        if prediction > 0:
            return self.pos_child
        else:
            return self.neg_child

    def predict(self, X):
        """
        make prediction
        :param x: 1-D array-like, represents a single feature vector
        :return: predicted label if self is the terminal node, else the score
        """
        if self.is_terminal:
            return self.positive_class

        if len(X.shape) < 2 or X.shape[0] == 1:
            return self._predict_single(X)

        return X.dot(self.weight) + self.bias

    def _predict_single(self, x):
        y = x.dot(self.weight.T) + self.bias
        return y

    # TODO: replace majority voting with prediction distribution
    def terminate(self, Y):
        c = Counter(Y.flatten())
        self.positive_class = c.most_common(1)[0][0]
        self.is_terminal = True

    # TODO: rewrite this stupid algorithm with cython
    def fit(self, X, Y):
        # randomly divide positive and negative classes
        pc = Y[random.randint(0, len(Y))]
        self.positive_class = pc

        xp = (X[Y == pc])[0]
        xn = X[Y != pc]
        if len(xn) == 0:
            return False
        xn = xn[0]

        w = (xp - xn)
        w /= np.linalg.norm(w)
        b = 1
        ep = self.err
        T = max(np.int(1 / ep ** 2), 1)
        idx = list(xrange(len(X)))
        # random.shuffle(idx)

        # SGD
        for t in xrange(T):
            if t % len(X) == 0:
                random.shuffle(idx)
            i = idx[t % len(X)]
            lx = 2 * (Y[i] == pc) - 1
            if (w.dot(X[i]) + b) * lx < 0:
                step_size = np.power(0.9, t * ep ** 2)
                w += step_size * X[i] * lx
                b += step_size * lx

        self.weight = w
        self.bias = b
        return True


class Tree(object):

    def __init__(self, max_depth, min_datasize, err):
        self.root = TreeNode(err, 0)
        self.max_depth = max_depth
        self.min_datasize = min_datasize
        self.err = err

    def fit(self, X, Y):
        queue = deque()
        queue.append((self.root, X, Y))

        # TODO: parallel to accelerate
        while len(queue) > 0:
            node, X, Y = queue.popleft()
            if len(X) <= self.min_datasize or node.depth >= self.max_depth:
                node.terminate(Y)
                continue
            if not node.fit(X, Y):
                node.terminate(Y)
                continue

            result = node.predict(X)
            X1, Y1 = X[result > 0], Y[result > 0]
            X2, Y2 = X[result <= 0], Y[result <= 0]
            if len(X1) == 0 or len(X2) == 0:
                node.terminate(Y)
                continue

            node.pos_child = TreeNode(self.err, node.depth + 1)
            node.neg_child = TreeNode(self.err, node.depth + 1)
            queue.append((node.pos_child, X1, Y1))
            queue.append((node.neg_child, X2, Y2))

    def predict(self, X):
        r = self.root
        if len(X.shape) < 2 or X.shape[0] == 1:
            return self._predict_single(X)

        result = []
        for x in X:
            result.append(self._predict_single(x))
        return np.array(result)

    def _predict_single(self, x):
        r = self.root
        if r.is_terminal:
            return r.predict(x)

        while not r.is_terminal:
            prediction = r.predict(x)
            r = r.direct(prediction)

        return r.predict(x)

class RandomForest(object):

    def __init__(self, max_depth=np.inf, min_datasize=10, err=0.01, forest_size=100):
        self.base_trees = []
        self.max_depth = max_depth
        self.min_datasize = min_datasize
        self.err = err
        self.forest_size = forest_size

    def fit(self, X, Y):

        # TODO: parallel to accelerate
        for i in xrange(self.forest_size):
            self.base_trees.append(Tree(max_depth=self.max_depth,
                                        min_datasize=self.min_datasize, err=self.err))
            self.base_trees[-1].fit(X, Y)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        if len(X.shape) < 2 or X.shape[0] == 1:
            return self._predict_single(X)
        else:
            Y = []
            for x in X:
                Y.append(self._predict_single(x))
            return np.array(Y)

    def _predict_single(self, x):
        c = Counter()
        for t in self.base_trees:
            c[t.predict(x)] += 1
        return c.most_common(1)[0][0]
