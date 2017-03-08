import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_prediction_on_data(x, y_pred, y_true):
    plt.plot(x,y_pred, color='red')
    plt.scatter(x,y_true, color='blue', alpha=0.3)
    plt.show()

def get_linreg_summary_sm(model):

    print (model.summary())

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.scatter(model.fittedvalues, model.outlier_test()['student_resid'],
                alpha=0.3)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Studentized residuals')
    plt.show()

def get_model_predictions_df(model_estimator, df, label, feature_list, dummy_list, index='Property_ID', loft_sample=False):

    print ("Running {} for label {}".format(model_estimator,label))
    dummy_df_list = get_dummy_dfs(df, dummy_list)
    X = prep_model_df(df, feature_list, dummy_df_list)
    X = X.drop(label, axis=1)
    y = df[label]

    if loft_sample:
        train_ind, test_ind = get_loftium_train_test_split(df)
        X_train = X[train_ind]
        X_test = X[test_ind]
        y_train = y[train_ind]
        y_test = y[test_ind]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y)

    print ("Splitting into {} training instances".format(len(X_train)))
    print (X_train.shape, X_test.shape)
    print ("========")
    print ("Running cross validation...")
    cv_mse = -get_cv_score(GradientBoostingRegressor(), X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_r2 = get_cv_score(GradientBoostingRegressor(), X_train, y_train, cv=5, scoring='r2')
    print ("CV MSE: {}, CV R2: {}".format(cv_mse, cv_r2))
    print ("========")
    print ("Training model...")
    y_pred, fitted_model = get_model_pred(model_estimator, X_train, X_test,y_train)
    mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)
    print ("Test MSE: {}, Test R2: {}".format(mse, r2))
    print ("Returned pred df with shape {}".format(y_pred.shape))

    Xout = df.copy()
    Xout['pred'] = fitted_model.predict(X)

    return Xout

def get_cv_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'):
    return np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring))

def get_loftium_train_test_split(df):
    insample_prop_ids = []
    with open('data/Loftium/insample_prop_ids.txt') as infile:
        for line in infile:
            insample_prop_ids.append(int(line.strip()))

    train_index = df.Property_ID.isin(insample_prop_ids)
    test_index = ~df.Property_ID.isin(insample_prop_ids)

    print (np.sum(train_index), np.sum(test_index))

    return train_index, test_index

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

def get_model_pred(model_estimator, X_train, X_test, y_train):
    model = model_estimator()
    model.fit(X_train, y_train)
    return model.predict(X_test), model
