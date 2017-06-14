"""
Nested 10-fold cross validation random forest regressor.
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

Common practice: smaller number of folds for the internal cross-validation (e.g. 2,3).


######################
To make predictions: simple cross-validation to determine optimal value(s) of the hyperparameter(s). Next, train the 
specified model on the entire training set (this is done automatically with GridSearchCV param refit=True. 
For the hold out sample instance asset return is predicted, which is subsequently used for the derivation of 
loss distribution.

note1: Question to Rui
How to handle different optimal value(s) for hyperparameter(s) due to different random data splits?

note2: GridSearchCV
If n_jobs was set to a value higher than one, the data is copied for each point in the grid (and not n_jobs times).
This is done for efficiency reasons if individual jobs take very little time, but may raise errors if the dataset is
large and not enough memory is available. A workaround in this case is to set pre_dispatch. Then, the memory is copied
only pre_dispatch many times. A reasonable value for pre_dispatch is 2 * n_jobs.

RESULT 1:
n_estimators = range(10, 30, 10)
max_features = np.linspace(0.2, 1, 5)
MSE: 208.763430 (1.431418)
RMSE: 14.448648 (1.196419)
Execution time: 756.761370

RESULT 2:
n_estimators = [10, 100, 400, 800, 1000]
max_features = np.linspace(0.2, 1, 5)
MSE: 197.772358 (0.000000)
RMSE: 14.063156 (0.000000)
Execution time: 5897.95219

#############################################################
RESULT 3: nested 10-fold cross validation (10x) | 10 fold outer-cv, 3 fold inner-cv
OS: Linux Ubuntu 16.04 LTS
AMI ID: ami-a8d2d7ce
Instance Type: c3.8xlarge
Availability zone: eu-west-1b
n_estimators = range(100, 600, 100)
max_features = np.linspace(0.2, 1, 5)
MSE: 198.728652 (0.912246)
RMSE: 14.097115 (0.955115)
Execution time: 8884.653687
################################################################



######################

"""

# Import libraries
import pandas as pd
import time
import os
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pickle import load

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42) # 42:The answer to life, the universe and everything.

# Load normalized data into dataframe
filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'resources/Data/SP500_data_Norm.sav')
spx = load(open(filename, 'rb'))

# Separate spx data into feature and response components
X = spx.iloc[:, 1:-1] # feature matrix
y = spx.iloc[:, -1]   # response vector

# Number of random trials
n_trials = 10
nested_scores = np.zeros(n_trials)

# Random forest model
model = RandomForestRegressor()
n_trees = range(100, 1100, 100)
n_estimators = range(100, 600, 100)
max_features = np.linspace(0.2, 1, 5)
param_grid = dict(max_features=max_features, n_estimators=n_estimators)

# Loop for each trial
# Note: for each trial one obtains a model with optimal hyperparameters. These may be different for each trial as
# each random split is different.
start_time = time.time()
for i in range(n_trials):
    inner_cv = KFold(n_splits=3, shuffle=True) # Common practice: smaller num. of folds for inner cross validation
    outer_cv = KFold(n_splits=10, shuffle=True) # Preferably 10 outer folds
    # Inner cross validation with hyperparameter optimization
    gsearch = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=inner_cv)
    # Fit the cross validated grid search on the data (gscv.fit(X, y))
    # Show best estimator
    #print(gsearch.best_params_)
    # Outer cross validation (use multiple runs of 10-fold cross validation with statistical significance tests for
    # meaningful comparison of different algorithms).
    nested_score = cross_val_score(gsearch, X, y, cv=outer_cv, scoring='neg_mean_squared_error', n_jobs=-1)
    nested_scores[i] = nested_score.mean()

# Calculate scores MSE, RMSE
mse_mean = -nested_scores.mean()  # fixed sign of MSE scores
mse_std = nested_scores.std()
mse_scores = "%s: %f (%f)" % ('MSE', mse_mean, mse_std)
# Convert from MSE to RMSE
rmse_mean = np.sqrt(mse_mean)
rmse_std = np.sqrt(mse_std)
rmse_scores = "%s: %f (%f)" % ('RMSE', rmse_mean, rmse_std)
# Print scores MSE, RMSE, execution time
print(mse_scores)
print(rmse_scores)
t_elapsed = "%s: %f" % ('Execution time', (time.time() - start_time))
print(t_elapsed)








