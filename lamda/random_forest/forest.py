__author__ = 'guoxy'

import numpy as np

# __all__ = ["RandomForest"]


class TreeNode(object):

    def __init__(self):
        pass

    @property
    def left_child(self):
        return None

    @property
    def right_child(self):
        return None

    @property
    def class_distr(self):
        return None

    @property
    def classifier(self):
        return None

    def classify(self, X):
        return None


def build_tree():
    return None


class RandomForest(object):

    def __init__(self, max_depth=np.inf, min_size=10, acc=0.1,
                 tree_size=20):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass