import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence, partial_dependence
from sklearn.base import clone

class AirbnKEY_Model(object):

    def __init__(self, estimator, conf_interval):
        self.estimator = estimator
        self.X_train = None
        self.y_train = None
        self.cv_score_ = None
        self.test_mse_ = None
        self.conf_interval = conf_interval
        self.cv_folds = None

    def fit(self, X_train, y_train):
        self.X_train=X_train
        self.y_train=y_train
        self.estimator.fit(X_train, y_train)
        return self.estimator

    def cv_mse(self,cv_folds=5):
        self.cv_folds = cv_folds
        self.cv_mse_ = np.mean(cross_val_score(self.estimator, self.X_train, self.y_train,
                                               cv=cv_folds, scoring='neg_mean_squared_error'))
        return -self.cv_mse_

    def predict(self,X):
        return self.estimator.predict(X)

    def test_mse(self,X,y):
        y_pred = self.predict(X)
        self.test_mse_ = mean_squared_error(y_pred, y)
        return self.test_mse_

    def predict_conf_interval(self, X_test, y_test):
        alpha = (1. - self.conf_interval)/2

        estimator_lower = clone(self.estimator)
        estimator_lower.set_params(alpha=alpha, loss='quantile')
        estimator_lower.fit(self.X_train,self.y_train)
        y_pred_lower = estimator_lower.predict(X_test)

        estimator_upper = clone(self.estimator)
        estimator_upper.set_params(alpha=1-alpha, loss='quantile')
        estimator_upper.fit(self.X_train,self.y_train)
        y_pred_upper = estimator_upper.predict(X_test)

        return y_pred_lower, y_pred_upper


def get_dummy_dfs(df, dummy_list):
    '''Convert dummy list to dummy dfs and append to df'''
    dummy_dfs=[]
    for dummy in dummy_list:
        dummy_dfs.append(pd.get_dummies(df[dummy]))
    return dummy_dfs

def prep_model_df(df, features, dummys):

    model_df = df[features]
    for dummy in dummys:
        model_df = pd.concat([model_df,dummy],axis=1)
    return model_df
