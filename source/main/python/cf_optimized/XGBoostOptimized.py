"""
the learning rate v in the range of [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1].
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
np.random.seed(42) # 42:The answer to life, the universe and everything.

# Load normalized data into dataframe
filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'resources/Data/SP500_data_Norm.sav')
spx = load(open(filename, 'rb'))

# Separate spx data into feature and response components
X = spx.iloc[:, 1:-1]  # feature matrix
y = spx.iloc[:, -1]    # response vector

# Number of random trials
n_trials = 10
nested_score_MAE = np.zeros(n_trials)
nested_score_MSE = np.zeros(n_trials)

# Extreme Gradient Boosting model
model = XGBRegressor()
##### Define parameter grid ######
v = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]


