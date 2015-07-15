# __author__ = 'guoxy'
from __future__ import division
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold
from forest import *

def k_fold_cv(classifier, X, Y, k, verbose=False, early_stop=None):
    """

    :param classifier:
    :param X:
    :param Y:
    :param k:
    :return: mean and standard variance of cv accuracy
    """
    skf = StratifiedKFold(Y, n_folds=k, shuffle=True)
    acc = []
    i = 0
    for train_idx, test_idx in skf:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        classifier.fit(X_train, Y_train)
        results = classifier.predict(X_test)
        acc.append((results == Y_test).sum() / len(Y_test))
        if verbose:
            print(" === CV round i === ".format(i))
            print(acc[-1])
            i += 1
        if early_stop is not None and early_stop == i:
            break
    acc = np.array(acc)
    return acc.mean(), acc.std()

class ForestTestCase(unittest.TestCase):

    def setUp(self):
        self.toy_X_train = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
        self.toy_Y_train = np.array([0, 0, 0, 1, 1, 1])
        self.toy_X_test = np.array([[-1, -1], [2, 2], [3, 2]])
        self.toy_Y_test = np.array([0, 1, 1])
        data = load_iris()
        self.X = data.data
        self.Y = data.target
        self.X, self.Y = shuffle(self.X, self.Y, random_state=0)

    def test_TreeNode(self):
        node = TreeNode(err=0.1, depth=1)
        node.fit(self.toy_X_train, self.toy_Y_train)
        result = node.predict(self.toy_X_test) > 0
        if node.positive_class == 0:
            result ^= 1
        assert_array_almost_equal(result, self.toy_Y_test)

    def test_Tree(self):
        tree1 = Tree(max_depth=np.inf, min_datasize=1, err=0.1)
        tree1.fit(self.toy_X_train, self.toy_Y_train)
        result = tree1.predict(self.toy_X_test)
        assert_array_almost_equal(np.array(result), self.toy_Y_test)

        tree2 = Tree(max_depth=np.inf, min_datasize=5, err=0.1)
        acc_mean, acc_std = k_fold_cv(tree2, self.X, self.Y, k=5)
        print("=== Single Tree ===")
        print(acc_mean)
        assert acc_mean >= 0.3

    def test_RandomForest(self):
        forest1 = RandomForest(max_depth=np.inf, min_datasize=1, err=0.1, forest_size=20)
        forest1.fit(self.toy_X_train, self.toy_Y_train)
        result = forest1.predict(self.toy_X_test)
        assert_array_almost_equal(np.array(result), self.toy_Y_test)

        forest2 = RandomForest(max_depth=np.inf, min_datasize=10, err=0.1, forest_size=100)
        acc_mean, acc_std = k_fold_cv(forest2, self.X, self.Y, k=5)
        print("=== Random Forest ===")
        print(acc_mean)
        assert acc_mean >= 0.3

if __name__ == '__main__':
    unittest.main()
