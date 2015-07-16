#__author__ = 'guoxy'
from __future__ import division
from collections import deque
from sklearn.externals.joblib import Parallel, delayed
import numpy as np
import sgd
import numpy.random as random

__all__ = ["RandomForest", "Tree", "TreeNode"]

# TODO: write full documents

class TreeNode(object):

    def __init__(self, err, depth):
        self.depth = depth
        self.err = err
        self.weight = None  # weight vector for internal linear classifier
        self.bias = None    # bias for internal linear classifier
        self.pos_child = None
        self.neg_child = None
        self.class_distr = None
        self.positive_class = self.negative_class = None
        self.is_terminal = False

    def __str__(self):
        s = "is_terminal: {}\n".format(self.is_terminal)
        s += "weight: {}\n".format(self.weight)
        s += "bias: {}\n".format(self.bias)
        s += "positive_class: {}\n".format(self.positive_class)
        s += "depth: {}\n".format(self.depth)
        s += "err: {}\n".format(self.err)
        return s

    @property
    def classifier(self):
        return self.weight, self.bias

    def direct(self, prediction):
        """
        Util func for iteratively making predictions.
        :param prediction: a real number which equals w^T * x+b at current node
        :return: next node
        should only be called in non-terminal nodes.
        """
        if prediction > 0:
            return self.pos_child
        else:
            return self.neg_child

    def predict(self, X):
        """
        making predictions for a bunch of instances
        :param x: mxn-shaped array-like, represents m feature vectors with dim n
        :return: predicted label distribution vector if current node is a terminal node,
                else the score for directing.
        """
        if self.is_terminal:
            return self.class_distr

        if len(X.shape) < 2 or X.shape[0] == 1:
            return self._predict_single(X)

        return X.dot(self.weight) + self.bias

    def _predict_single(self, x):
        """
        making prediction for a single instance
        :param x: 1-D array-like, represents a single feature vector
        :return: w^T * x + b
        should only be called in non-terminal nodes
        """
        if self.weight is None:
            print("Fuck")
            print(self)
            exit(1)
        y = x.dot(self.weight) + self.bias
        return y

    def terminate(self, Y, y_range):
        """
        making current node a terminal node.
        :param Y: 1-D array-like with int64 elems, the remained labels in current node
        :param y_range: max(Y)
        :return: None
        The class distribution will be calculated only in terminal node
        """
        elem, occur = np.unique(Y, return_counts=True)
        self.positive_class = elem[-1]
        self.is_terminal = True
        self.class_distr = np.zeros(y_range + 1)
        for i, e in enumerate(elem):
            self.class_distr[e] = occur[i]
        self.class_distr /= self.class_distr.sum()

    # DONE: np.ndarray is passed by reference to cythonized func
    def fit(self, X, Y):
        """
        train the internal linear classifier
        :param X: mxn-shaped numpy.ndarray with float64 elems,
                  each row represents a feature vector.
        :param Y: 1-D numpy.ndarray with int64 elems,
                  represents the corresponding labels
        :return: None
        """
        self.weight, self.bias, self.positive_class = sgd.sgd(X, Y, self.err)

        if self.weight is None or \
            self.bias is None or \
                self.positive_class is None:
            return False
        return True


class Tree(object):

    def __init__(self, max_depth, min_datasize, err, y_range=None):
        """
        modified decision tree classifer
        :param max_depth: maximum tree depth
        :param min_datasize: minimum datasize for each node
        :param err: the error rate for training each node
        :param y_range: the possible max class labels
        :return: None
        """
        self.root = TreeNode(err, 0)
        self.max_depth = max_depth
        self.min_datasize = min_datasize
        self.err = err
        self.y_range = y_range

    def fit(self, X, Y):
        queue = deque()
        if self.y_range is None:
            self.y_range = np.max(Y)
        idxs = np.array(xrange(len(X)))
        queue.append((self.root, idxs))

        # TODO: parallelizing to accelerate
        while len(queue) > 0:
            node, index = queue.popleft()
            X_curr = X[index]
            Y_curr = Y[index]
            if len(index) <= self.min_datasize or node.depth >= self.max_depth:
                node.terminate(Y_curr, self.y_range)
                continue
            if not node.fit(X_curr, Y_curr):
                node.terminate(Y_curr, self.y_range)
                continue

            result = node.predict(X_curr)
            idx_pos = index[result.flatten() > 0]
            idx_neg = index[result.flatten() <= 0]
            if len(idx_pos) == 0 or len(idx_neg) == 0:
                node.terminate(Y_curr, self.y_range)
                continue

            node.pos_child = TreeNode(self.err, node.depth + 1)
            node.neg_child = TreeNode(self.err, node.depth + 1)
            queue.append((node.pos_child, idx_pos))
            queue.append((node.neg_child, idx_neg))

    def predict(self, X):
        """
        making predictions
        :param X: mxn-shaped array-like, each row represents a single instance
        :return: 2-D array-like, each row represents a predicted label distribution
        """
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


def _parallel_build_helper(tree, X, Y):
    tree.fit(X, Y)
    return tree


class RandomForest(object):

    def __init__(self, max_depth=np.inf, min_datasize=10, err=0.01, forest_size=20):
        """
        Random forest classifier
        :param max_depth: maximum depth for each tree classifiers (default inf)
        :param min_datasize: minimum data size for each tree classifier (default 3)
        :param err: error rate for training base linear classifier at each tree node (default 0.01)
        :param forest_size: number of trees that will be build
        :return: None
        """
        self.base_trees = []
        self.max_depth = max_depth
        self.min_datasize = min_datasize
        self.err = err
        self.forest_size = forest_size
        self.y_range = None

    def _transform_input(self, Y):
        # TODO: check input
        pass

    def fit(self, X, Y):
        """
        Fit training set
        :param X: mxn-shaped array-like, each row represents a single instance
        :param Y: 1-D array-like with length m, contains corresponding class labels
        """
        self.y_range = np.max(Y)
        for i in xrange(self.forest_size):
            self.base_trees.append(Tree(max_depth=self.max_depth,
                                        min_datasize=self.min_datasize,
                                        err=self.err,
                                        y_range=self.y_range))
        # TODO: ensure that mmap has taken effect
        self.base_trees = Parallel(n_jobs=-1, max_nbytes='100M')(
            delayed(_parallel_build_helper)(t, X, Y)
        for t in self.base_trees)

    def predict(self, X):
        """
        making predictions
        :param X: mxn-shaped array-like, each row represents a single instance
        :return: 1-D np.ndarray with length n, contains predicted labels
        """
        # TODO: parallelize the prediction process
        if len(X.shape) < 2 or X.shape[0] == 1:
            return self._predict_single(X)
        else:
            Y = []
            for x in X:
                Y.append(self._predict_single(x))
            return np.array(Y)

    def _predict_single(self, x):
        prediction = np.zeros(self.y_range + 1)
        for t in self.base_trees:
            prediction += t.predict(x)
        return np.argmax(prediction)
