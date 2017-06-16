
# Imports
import os
from pickle import dump
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor



#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42) # 42:The answer to life, the universe and everything.

# Load SPX dataframe
picklename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources/Data/SP500_data_Norm.sav')
spx = load(open(picklename, 'rb'))


# Separate spx data into feature matrix and response vector
X = spx.iloc[:, 1:-1] # feature matrix
y = spx.iloc[:, -1]   # response vector

w_features = np.zeros(X.shape[1]) # vector with feature importance weights

rf = RandomForestRegressor()
rf.fit(X, y)
w_features = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(w_features)[::-1] # Sort feature indices in decreasing feature importance weight

# Mean decrease impurity
""" Every node in the decision trees is a condition on a single feature, designed to split
the dataset into two so that similar response values end up in the same set. The measure 
based on which the (locally) optimal condition is chosen is called impurity. 
For regression trees it is variance (mean squared error). Thus when training a tree, 
it can be computed how much each feature decreases the weighted impurity in a tree. 
For a forest, the impurity decrease from each feature can be averaged and the features 
are ranked according to this measure.
"""

print("Feature Importance Ranking:")
total = 0.00
feature_names = map(lambda x: x, spx)[1:-1]

for i in range(X.shape[1]):
    print("%d. %s (%f)" % (i + 1, feature_names[indices[i]], w_features[indices[i]]))


""" Verify sum of feature importance weights add up to one: Displayed are the relative importance
of each feature, hence the sum of all feature importance weights should be equal to one."""
#print(sum(w_features))


# Plot the feature importance weights of the random forest
# The red bars are the feature importance weights of the random forest, along with their inter-trees variability.
plt.figure()
plt.title("Feature Importance Weights")
plt.bar(range(X.shape[1]), w_features[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
#plt.show()



# Save vector with feature importance weights using pickle
picklename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'resources/Data/w_featuresSP500.sav')
dump(w_features, open(picklename, 'wb'))





