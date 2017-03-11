import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence, partial_dependence
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

    return Xout, X_train, fitted_model

def plot_feature_importances(fitted_model, X_train):
    features_names = X_train.columns
    feature_importances = 100*fitted_model.feature_importances_ / np.sum(fitted_model.feature_importances_)
    feature_importances, feature_names, feature_idxs = zip(*sorted(zip(feature_importances, features_names, range(len(features_names)))))
    fig, ax = plt.subplots(1,figsize=(14,10))

    width = 0.8

    idx = np.arange(len(features_names))
    ax.barh(idx, feature_importances, align='center')
    plt.yticks(idx, feature_names)

    plt.title("Feature Importances in Gradient Booster")
    plt.xlabel('Relative Importance of Feature', fontsize=14)
    plt.ylabel('Feature Name', fontsize=14)
    plt.show()
    #plt.savefig('plots/feature-importances.png', bbox_inches='tight')

def plot_partial_dependency_plots(fitted_model, X_train, n_col=3, n_row=3, xlabel_height=20):
    N_COLS = n_col
    N_ROWS = n_row
    features_names = X_train.columns
    feature_importances = 100*fitted_model.feature_importances_ / np.sum(fitted_model.feature_importances_)
    feature_importances, feature_names, feature_idxs = zip(*sorted(zip(feature_importances, features_names, range(len(features_names)))))

    fimportances = list(reversed(feature_importances))
    fnames = list(reversed(feature_names))

    pd_plots = [partial_dependence(fitted_model, target_feature, X=X_train, grid_resolution=50)
                for target_feature in feature_idxs]
    pd_plots = list(reversed(zip([pdp[0][0] for pdp in pd_plots], [pdp[1][0] for pdp in pd_plots])))

    fig, axes = plt.subplots(nrows=N_ROWS, ncols=N_COLS, sharey=True,
                             figsize=(12.0, 12.0))

    for i, (y_axis, x_axis) in enumerate(pd_plots[0:(N_ROWS*N_COLS)]):
        ax = axes[i/N_COLS, i%N_COLS]
        ax.plot(x_axis, y_axis, color="purple")
        ax.set_xlim([np.min(x_axis), np.max(x_axis)])
        text_x_pos = np.min(x_axis) + 0.05*(np.max(x_axis) - np.min(x_axis))
        ax.text(text_x_pos, xlabel_height,
                "Feature Importance " + str(round(fimportances[i], )),
                fontsize=12, alpha=0.5)
        ax.set_xlabel(fnames[i])

    plt.suptitle("Partial Dependence Plots (Ordered by Feature Importance)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
    #plt.savefig('plots/patial-dependence-plots.png', bbox_inches='tight')

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
