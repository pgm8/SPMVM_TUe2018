""" Compare algorithms with default settings.
H0: Accuracy is higher for asset return predictions based on random forests (cf. extreme gradient boosted trees and 
feature weighted k-nearest neighbors) with default hyperparameter settings. 

No optimization of hyperparameter(s), therefore repeated simple cross validation is sufficient for providing an 
answer to the research hypothesis.

RESULT:
KNN(mse): 344.351417 (3.268540)
KNN(rmse): 18.556708 (1.807910)
RF(mse): 220.690670 (2.627905)
RF(rmse): 14.855661 (1.621082)
XGB(mse): 211.245536 (1.764457) // GB(mse): 212.485820 (1.589273)
XGB(rmse): 14.534288 (1.328329) // GB(rmse): 14.576893 (1.260664)
Execution time: 86.443708

"""

# Import libraries
import pandas as pd
import numpy as np
import time
import os.path
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRegressor

#  Set seed for pseudorandom number generator. This allows us to reproduce the results from our script.
np.random.seed(42) # 42:The answer to life, the universe and everything.

# Load normalized data into dataframe
filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources/SP500_data_Norm.csv')
spx = pd.read_csv(filename)
spx = spx.drop('Unnamed: 0', axis=1)

# Separate spx data into feature and response components
X = spx.iloc[:, 1:-1] # feature matrix
y = spx.iloc[:, -1]   # response vector

# Prepare models
models = []
models.append(('KNN', KNeighborsRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('GB', GradientBoostingRegressor()))
#models.append(('XGB', XGBRegressor()))

# Evaluate each model in turn
names = []
scoring = 'neg_mean_squared_error'
n_trials = 10
scores = np.zeros(n_trials)

start_time = time.time()
for name, model in models:
    for i in range(n_trials):
        kfold = KFold(n_splits=10, shuffle=True)
        cv_simple = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
        scores[i] = cv_simple.mean()
        names.append(name)
    # Calculate scores MSE, RMSE
    mse_mean = -scores.mean()
    mse_std = scores.std()
    mse_scores = "%s: %f (%f)" % (name+'(mse)', mse_mean, mse_std)
    # Convert from MSE to RMSE
    rmse_mean = np.sqrt(mse_mean)
    rmse_std = np.sqrt(mse_std)
    rmse_scores = "%s: %f (%f)" % (name+'(rmse)', rmse_mean, rmse_std)
    # Print scores MSE, RMSE, execution time
    print(mse_scores)
    print(rmse_scores)
print("%s: %f" % ('Execution time', (time.time() - start_time)))





