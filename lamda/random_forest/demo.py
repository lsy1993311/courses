__author__ = 'guoxy'
import timeit
s = """\
from forest import RandomForest
from test_forest import k_fold_cv
from sklearn.datasets import load_digits, fetch_mldata

mnist = fetch_mldata('mnist')
Xm = mnist.data.toarray()
Ym = mnist.target
# digits = load_digits()
# Xd = digits.data
# Yd = digits.target
rf = RandomForest(forest_size=20, err=0.01)

# m, s = k_fold_cv(rf, Xd, Yd, k=5, verbose=True, early_stop=1)
m, s = k_fold_cv(rf, Xm, Ym, k=5, verbose=True, early_stop=1)
print("mean acc={}, std={}".format(m, s))
"""
print(timeit.timeit(stmt=s, number=1))