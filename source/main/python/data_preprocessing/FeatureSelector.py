

import numpy as np
import os
import matplotlib.pyplot as plt
from pickle import load
from sklearn.decomposition import PCA

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42)  # 42:The answer to life, the universe and everything.

# Load SPX dataframe
picklename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'resources/Data_mw_true/SP500_data_Norm.pkl')
spx = load(open(picklename, 'rb'))

# Separate spx data into feature matrix and response vector
X = spx.iloc[:, 1:-1]  # feature matrix
y = spx.iloc[:, -1]    # response vector

# PCA for feature importance decomposition
pca = PCA()
decomp = pca.fit(X)

# Figure
plt.figure(1)
plt.axes([.2, .2, .7, .7])
plt.plot(decomp.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio')
plt.show()

# Component information
print("Explained Variance: %s") % (decomp.explained_variance_ratio_)

