"""
the learning rate v in the range of [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1].


####### Results ############
n_trials = 2, cv_inner = cv_outer = 3
n_estimators = range(100, 600, 100)
learning_rate = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
MAE: 110.821831 (0.000884)
RMSE: 117.387292 (0.000353)
Execution time: 580.671633


"""

# Import libraries
import time
import os
from pickle import load
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from MultiScorer import MultiScorer

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42)  # 42:The answer to life, the universe and everything.

# Load normalized data into dataframe
filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'resources/Data_mw_true/SP500_data_Norm.pkl')
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
##### Define parameter grid ######
n_estimators = range(100, 600, 100)
learning_rate = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate)


# Loop for each trial
# Note: for each trial one obtains a model with optimal hyperparameters. These may be different for each trial as
# each random split is different.
start_time = time.time()
for i in range(n_trials):
    scoring = MultiScorer({
        'MAE': (mean_absolute_error, {}),
        'MSE': (mean_squared_error, {})
    })
    inner_cv = KFold(n_splits=3, shuffle=True)  # Common practice: smaller num. of folds for inner cross validation
    outer_cv = KFold(n_splits=3, shuffle=True)  # Preferably 10 outer folds
    # Inner cross validation with hyperparameter optimization
    gsearch = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=-1, cv=inner_cv)
    # Outer cross validation (use multiple runs of 10-fold cross validation with statistical significance tests for
    # meaningful comparison of different algorithms).
    nested_score = cross_val_score(gsearch, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)
    nested_score_MAE[i] = np.average(scoring.get_results('MAE'))
    nested_score_MSE[i] = np.average(scoring.get_results('MSE'))

# Calculate scores MAE, MSE
print("%s: %4f (%4f)" % ('MAE', np.average(nested_score_MAE), np.std(nested_score_MAE)))
print("%s: %4f (%4f)" % ('RMSE', np.average(np.sqrt(nested_score_MSE)),
                         np.std(np.sqrt(nested_score_MSE))))
t_elapsed = "%s: %f" % ('Execution time', (time.time() - start_time))
print(t_elapsed)


