# Custom Modules
import data
from ml import StackingAveragedModels
#pandas
import pandas as pd
import numpy as np
# sklearn
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
# xgboost
import xgboost as xgb
# python
import logging

# Setup logging
logger = logging.getLogger()
handler = logging.FileHandler('/output/training.log')
formatter = logging.Formatter(
        '%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

train_X, train_y, test_X = data.get_data()

MODELS = [
    ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3),
    KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
    GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
        max_depth=4, max_features='sqrt',
        min_samples_leaf=15, min_samples_split=10, 
        loss='huber', random_state =5),
    xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
        learning_rate=0.05, max_depth=3, 
        min_child_weight=1.7817, n_estimators=2200,
        reg_alpha=0.4640, reg_lambda=0.8571,
        subsample=0.5213, silent=1,
        random_state =7, nthread = -1),
]

# Predict
clf = StackingAveragedModels(MODELS, Lasso(alpha =0.0005, random_state=1))
clf.fit(train_X, train_y)
score = metrics.mean_squared_error(train_y, clf.predict(train_X))
logger.info("Score on Training Set: ")
logger.info(score)
test_y_pred = clf.predict(test_X)

# Prepare submission
sub = pd.DataFrame()
test = pd.read_csv('/input/test.csv')
sub['Id'] = test['Id']
sub['SalePrice'] = np.expm1(test_y_pred)
sub.to_csv('/output/submission.csv',index=False)