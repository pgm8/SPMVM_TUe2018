"""


The alternative strategy deployed for hyperparameter optimization:

1. Choose a relatively high learning rate. Our default choice is a value of 0.1.
2. Determine the optimum number of trees for this learning rate with early stopping to avoid overfitting.
3. Tune tree-specific parameters for the decided learning rate and number of trees. We may evaluate
a grid of parameter pairs: the maximum tree depth is grid searched in the range [2, 4, 6, 8, 10] and
the subsample ratio of columns for each split is grid searched in the range [0.2, 0.4, 0.6, 0.8, 1.0].
4. Lower the learning rate and increase the number of trees proportionally to get more robust models.





"""

# Import libraries
import time
import os
from pickle import load
import numpy as np
import scipy.stats
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from MultiScorer import MultiScorer

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42)  # 42:The answer to life, the universe and everything.

# Load normalized data into dataframe
filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'resources/Data/SP500_data_Norm.pkl')
spx = load(open(filename, 'rb'))

# Separate spx data into feature and response components
X = spx.iloc[:, 1:-1]  # feature matrix
y = spx.iloc[:, -1]    # response vector

# Number of random trials
n_trials = 2
nested_score_MAE = np.zeros(n_trials)
nested_score_MSE = np.zeros(n_trials)

# Extreme Gradient Boosting model
model = XGBRegressor()
# Define parameter grid
n_estimators = range(100, 600, 100)
learning_rate = 0.1
max_depth = range(2, 10, 2)
colsample_bylevel = range(0.2, 1, 0.2)
param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate)


# Loop for each trial
# Note: for each trial one obtains a model with optimal hyperparameters. These may be different for each trial as
# each random split is different.
start_time = time.time()
for i in range(n_trials):
    """learning_rate v = 0.1, early stopping time to establish number of rounds used for gridsearch 
    parameter pairs tree depth and column sub sampling ratio by level (can be different
     number of rounds for different random partitions 
    """

    inner_cv = KFold(n_splits=3, shuffle=True)  # Common practice: smaller num. of folds for inner cross validation
    outer_cv = KFold(n_splits=3, shuffle=True)  # Preferably 10 outer folds
    # Inner cross validation with hyperparameter optimization
    # gridsearch
    gsearch = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=inner_cv)
    # Outer cross validation (use multiple runs of 10-fold cross validation with statistical significance tests for
    # meaningful comparison of different algorithms).
    nested_score = cross_val_score(gsearch, X, y, cv=outer_cv, scoring='neg_mean_absolute_error', n_jobs=1)

    # cross val score


# Calculate scores MAE, MSE
print("%s: %4f (%4f)" % ('MAE', np.average(nested_score_MAE), np.std(nested_score_MAE)))
print("%s: %4f (%4f)" % ('RMSE', np.average(np.sqrt(nested_score_MSE)),
                         np.std(np.sqrt(nested_score_MSE))))
t_elapsed = "%s: %f" % ('Execution time', (time.time() - start_time))
print(t_elapsed)


