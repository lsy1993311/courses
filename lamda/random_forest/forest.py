#__author__ = 'guoxy'
from __future__ import division
from collections import deque, Counter
from sklearn.externals.joblib import Parallel, delayed
import pathos.multiprocessing as mp
import numpy as np
import sgd
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
            return self.class_distr

        if len(X.shape) < 2 or X.shape[0] == 1:
            return self._predict_single(X)

        return X.dot(self.weight) + self.bias

    def _predict_single(self, x):
        if self.weight is None:
            print("Fuck")
            print(self)
            exit(1)
        y = x.dot(self.weight) + self.bias
        return y

    # TODO: replace majority voting with prediction distribution
    def terminate(self, Y, y_range):
        c = Counter(Y.flatten())
        self.positive_class = c.most_common(1)[0][0]
        self.is_terminal = True
        self.class_distr = np.zeros(y_range + 1)
        for k, v in c:
            self.class_distr[k] = v
        self.class_distr /= self.class_distr.sum()

    # TODO: rewrite this stupid algorithm with cython
    def fit(self, X, Y):
        self.weight, self.bias, self.positive_class = sgd.sgd(X, Y, self.err)

        if self.weight is None or \
            self.bias is None or \
                self.positive_class is None:
            return False
        return True


class Tree(object):

    def __init__(self, max_depth, min_datasize, err, y_range=None):
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

        # TODO: parallel to accelerate
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


def _parallel_build_helper(tree, X, Y):
    tree.fit(X, Y)
    return tree

# def _parallel_predict_helper(tree, x):
#     return tree.predict(x)


def _parallel_build_tree(X, Y, n_trees, **tree_paras):
    pool_size = mp.cpu_count()
    pool = mp.Pool(pool_size * 2)
    trees = []
    for i in xrange(n_trees):
        trees.append(Tree(**tree_paras))
    trees = pool.map(lambda x: _parallel_build_helper(x, X, Y), trees)
    pool.close()
    pool.join()
    return trees


class RandomForest(object):

    def __init__(self, max_depth=np.inf, min_datasize=10, err=0.01, forest_size=20):
        self.base_trees = []
        self.max_depth = max_depth
        self.min_datasize = min_datasize
        self.err = err
        self.forest_size = forest_size
        self.y_range = None

    def _transform_input(self, Y):
        pass

    def fit(self, X, Y):
        self.y_range = np.max(Y)

        # for i in xrange(self.forest_size):
        #     self.base_trees.append(Tree(max_depth=self.max_depth,
        #                                 min_datasize=self.min_datasize,
        #                                 err=self.err,
        #                                 y_range=self.y_range))
        #     # self.base_trees[-1].fit(X, Y)
        self.base_trees = _parallel_build_tree(X, Y, self.forest_size,  # followed by tree paras
                                               max_depth=self.max_depth,
                                               min_datasize=self.min_datasize,
                                               err=self.err,
                                               y_range=self.y_range)
        # self.base_trees = Parallel(n_jobs=-1, max_nbytes='100M')(
        #     delayed(_parallel_build_helper)(t, X, Y)
        # for t in self.base_trees)

    def predict(self, X):
        """

        :param X:
        :return:
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
        # c = Counter()
        prediction = np.zeros(self.y_range + 1)
        for t in self.base_trees:
            prediction += t.predict(x)
            # c[t.predict(x)] += 1
        # return c.most_common(1)[0][0]
        return np.argmax(prediction)
