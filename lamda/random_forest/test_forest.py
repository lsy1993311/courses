# __author__ = 'guoxy'
# testing file for forest.py
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

    :param classifier: classifer with a 'fit' method for training and a 'predict' method for prediction
    :param X: mxn-shaped array-like, each row represents a single instance
    :param Y: 1-D array-like with length m, which contains corresponding labels
    :param k: fold num
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
            print(" === CV round {} === ".format(i + 1))
            print("Validation accuracy: {0:7.4}".format(acc[-1]))
            i += 1
        if early_stop is not None and early_stop == i:
            break
    acc = np.array(acc)
    return acc.mean(), acc.std()

def assess_estimator(estimator, train_data, train_label,
                     test_data, test_label, grader):
    estimator.fit(train_data, train_label)
    prediction = estimator.predict(test_data)
    grader(prediction, test_label)

class ForestTestCase(unittest.TestCase):

    def setUp(self):
        self.toy_X_train = np.array([
            [-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]
        ]).astype(np.float64, copy=False)
        self.toy_Y_train = np.array([
            0, 0, 0, 1, 1, 1
        ]).astype(np.int64, copy=False)
        self.toy_X_test = np.array([
            [-1, -1], [2, 2], [3, 2]
        ]).astype(np.float64, copy=False)
        self.toy_Y_test = np.array([
            0, 1, 1
        ]).astype(np.int64, copy=False)

        data = load_iris()
        self.X = data.data.astype(np.float64, copy=False)
        self.Y = data.target.astype(np.int64, copy=False)
        self.X, self.Y = shuffle(self.X, self.Y, random_state=0)

    def test_TreeNode(self):
        node = TreeNode(err=0.1, depth=1)
        node.fit(self.toy_X_train, self.toy_Y_train)
        result = node.predict(self.toy_X_test) > 0
        if node.positive_class == 0:
            result ^= 1
        assert_array_almost_equal(result, self.toy_Y_test)

    def test_Tree(self):
        tree1 = Tree(max_depth=np.inf, min_datasize=1,
                     err=0.1, final='distribution')
        assess_estimator(estimator=tree1,
                         train_data=self.toy_X_train,
                         train_label=self.toy_Y_train,
                         test_data=self.toy_X_test,
                         test_label=self.toy_Y_test,
                         grader=lambda p, y:
                         assert_array_almost_equal(np.argmax(p, axis=1), y))

        tree2 = Tree(max_depth=np.inf, min_datasize=1,
                     err=0.1, final='single')
        assess_estimator(estimator=tree2,
                         train_data=self.toy_X_train,
                         train_label=self.toy_Y_train,
                         test_data=self.toy_X_test,
                         test_label=self.toy_Y_test,
                         grader=assert_array_almost_equal)

    def test_RandomForest(self):
        forest1 = RandomForest(max_depth=np.inf, min_datasize=1,
                               err=0.1, forest_size=20, final='single')
        assess_estimator(estimator=forest1,
                         train_data=self.toy_X_train,
                         train_label=self.toy_Y_train,
                         test_data=self.toy_X_test,
                         test_label=self.toy_Y_test,
                         grader=assert_array_almost_equal)

        forest2 = RandomForest(max_depth=np.inf, min_datasize=1,
                               err=0.1, forest_size=20, final='distribution')
        assess_estimator(estimator=forest2,
                         train_data=self.toy_X_train,
                         train_label=self.toy_Y_train,
                         test_data=self.toy_X_test,
                         test_label=self.toy_Y_test,
                         grader=assert_array_almost_equal)

        forest3 = RandomForest(max_depth=np.inf, min_datasize=10,
                               err=0.1, forest_size=20, final='single')
        acc_mean, acc_std = k_fold_cv(forest3, self.X, self.Y,
                                      k=5, verbose=True)
        print(acc_mean)
        assert acc_mean >= 0.9

        forest4 = RandomForest(max_depth=np.inf, min_datasize=10,
                               err=0.1, forest_size=20, final='distribution')
        acc_mean, acc_std = k_fold_cv(forest4, self.X, self.Y,
                                      k=5, verbose=True)
        print(acc_mean)
        assert acc_mean >= 0.9

if __name__ == '__main__':
    unittest.main()
