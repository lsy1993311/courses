# __author__ = 'guoxy'
from __future__ import division
import numpy as np
from time import time
from forest import RandomForest
from test_forest import k_fold_cv
from sklearn.datasets import load_iris, load_digits, fetch_mldata
from scipy.sparse import issparse

print("===============================================")
print("This is a demo for the RandomForest classifier.")
print("Choose a dataset to fit: ")
print("""\
1. iris:
    =================   ==============
    Classes                          3
    Samples total                  150
    Dimensionality                   4
    =================   ==============
2. digits:
    =================   ==============
    Classes                         10
    Samples total                 1797
    Dimensionality                  64
    =================   ==============
3. mnist :
    =================   ==============
    Classes                         10
    Samples total                70000
    Dimensionality                 780
    =================   ==============
    WARNING: the mnist dataset may be downloaded from
     internet, and you need a available network connection
""")
datasets_name = {'1': load_iris,
                 '2': load_digits,
                 '3': lambda : fetch_mldata('mnist')}
nchosen = None
while nchosen not in datasets_name:
    nchosen = raw_input("enter a number to choose: ")

print("===== getting datasets ... ====")
D = datasets_name[nchosen]()
print("Done. ")
X = D.data
if issparse(X):
    X = X.toarray()
Y = D.target
if issparse(Y):
    Y = Y.toarray()
X = X.astype(np.float64, copy=False)
Y = Y.astype(np.int64, copy=False)

print("==== Setting forest's parameter ====")
normalize = int(raw_input("whether to normalize data to range [0, 1]"
                          "(1 for yes, 0 for no, default 0)") or 0)
normalize = (normalize == 1)
if normalize:
    X /= X.max()
fst_size = int(raw_input("forest size (default 20): ") or 20)
mindt_size = int(raw_input("minimum data size at each node (default 3): ") or 3)
try:
    maxdp = int(raw_input("max tree depth (default inf): "))
except ValueError:
    maxdp = np.inf
max_err = float(raw_input("max error rate for each node (default 0.01): ") or 0.01)
rf = RandomForest(forest_size=fst_size, err=max_err,
                  min_datasize=mindt_size, max_depth=maxdp)

print("==== Training settings ====")
fold_num = int(raw_input("fold number for cross-validation (default 5):") or 5)
stop_after = int(raw_input("whether stopping after the first fold of cv (default 1):") or 1)
stop_after = max(stop_after, 1)
verbose = int(raw_input("""whether printing validation accuracy for each fold"
 (1 for yes, 0 for no, default 1):""") or 1)
verbose = (verbose == 1)

print("==== Begin training and validation ====")
t_start = time()
m, s = k_fold_cv(rf, X, Y, k=fold_num, verbose=verbose, early_stop=stop_after)
t_elapsed = time() - t_start
print("==== Final stats ====")
print("mean acc={0:7.4f}, std={1:7.4f}".format(m, s))
print("Total elapsed time: {0:7.3f} s".format(t_elapsed))
