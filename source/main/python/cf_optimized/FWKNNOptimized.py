"""
Nested 10-fold cross validation feature weighted knn regressor with inverse distance weighting.
Inner cross validation is applied to determine the optimal value(s) of the hyperparameter(s) for each fold
of the outer cross validation used to obtain the final performance estimate for the learning algorithm.

An outer cross validation is used to obtain an unbiased performance evaluation of the model selected by
the inner cross validation. We are interested in the performance of the learing method, not necessarily in the
performance of a specific model. Nested cross validation does not output a model, only scores.



Expensive process:
Grid search for two hyperparamaters and a 10x10 grid, 100 inner cross validations are needed, and this must be done
for each fold of the outer cross validation. Assuming k=10 for inner and outer cross validation, the learning algorithm
is run 10x10x100 = 10 000 times. And then we might want to repeat the outer cross validation 10 times to obtain
reliable final performance.

Common practice: smaller number of folds for the internal cross-validation (e.g. 2,3). """

# Import libraries
import time
import os.path, sys
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from pickle import load



#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42) # 42:The answer to life, the universe and everything.

# Load normalized data into dataframe
filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'resources/Data/SP500_data_Norm.sav')
spx = load(open(filename, 'rb'))
# Load feature importance vector
picklename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'resources/Data/w_featuresSP500.sav')
w = load(open(picklename, 'rb'))

# Separate spx data into feature and response components
X = spx.iloc[:, 1:-1] # feature matrix
y = spx.iloc[:, -1]   # response vector

# Number of random trials
n_trials = 2
nested_scores = np.zeros(n_trials)

# FWKNN model
model = KNeighborsRegressor(weights='distance', algorithm='brute', metric='wminkowski', p=2, metric_params={'w': w})
n_neighbors = range(1, 2, 1)
param_grid = dict(n_neighbors=n_neighbors)
"""
# Loop for each trial
# Note: for each trial one obtains a model with optimal hyperparameters. These may be different for each trial as
# each random split is different.
start_time = time.time()
for i in range(n_trials):
    inner_cv = KFold(n_splits=2, shuffle=True) # Common practice: smaller num. of folds for inner cross validation
    outer_cv = KFold(n_splits=2, shuffle=True) # Preferably 10 outer folds
    # Inner cross validation with hyperparameter optimization
    gsearch = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=inner_cv)
    # Outer cross validation (use multiple runs of 10-fold cross validation with statistical significance tests for
    # meaningful comparison of different algorithms).
    nested_score = cross_val_score(gsearch, X, y, cv=outer_cv, scoring='neg_mean_squared_error', n_jobs=-1)
    nested_scores[i] = nested_score.mean()

# Calculate scores MSE, RMSE
mse_mean = -nested_scores.mean()  # fixed sign of MSE scores
mse_std = nested_scores.std()
mse_scores = "%s: %4f (%4f)" % ('MSE', mse_mean, mse_std)
# Convert from MSE to RMSE
rmse_mean = np.sqrt(mse_mean)
rmse_std = np.sqrt(mse_std)
rmse_scores = "%s: %4f (%4f)" % ('RMSE', rmse_mean, rmse_std)
# Print scores MSE, RMSE, execution time
print(mse_scores)
print(rmse_scores)
t_elapsed = "%s: %f" % ('Execution time', (time.time() - start_time))
print(t_elapsed)


# neg_mean_squared_error
# neg_mean_absolute_error

"""