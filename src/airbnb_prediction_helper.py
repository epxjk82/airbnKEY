import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence, partial_dependence
from sklearn.base import clone
import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_prediction_on_data(x, y_pred, y_true):
    """Plot predictions on top of actuals"""

    plt.plot(x,y_pred, color='red')
    plt.scatter(x,y_true, color='blue', alpha=0.3)
    plt.show()

def get_linreg_summary_sm(model):
    """Prints linear regression summary and plots from statsmodels

    Parameters
    ----------
    model: fitted statsmodel
        A pre-fitted model

    Returns
    -------
    None
    """

    print (model.summary())

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.scatter(model.fittedvalues, model.outlier_test()['student_resid'],
                alpha=0.3)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Studentized residuals')
    plt.show()

def get_model_confidence_interval(fitted_model, X_train, y_train, conf_interval=0.95):
    """Fits GradientBoostingRegressor models for upper and lower bounds of confidence interval

    Parameters
    ----------
    fitted_model : GradientBoostingRegressor model class
        A GradientBoostingRegressor model that has already been fitted on training data
    X_train : array-like
        The training data that was used to fit fitted_model
    y_train : array-like
        The labeled data that was used to fit fitted_model
    conf_interval : float
        The confidence interval used to specify alpha in upper and lower bounds

    Returns
    -------
    GradientBoostingRegressor model class (upper bound)
    GradientBoostingRegressor model class (lower bound)
    """
    alpha = (1-conf_interval)/2

    model_CI_upper = clone(fitted_model)
    model_CI_upper.set_params(alpha=1-alpha, loss='quantile')
    model_CI_upper.fit(X_train,y_train)

    model_CI_lower = clone(fitted_model)
    model_CI_lower.set_params(alpha=alpha, loss='quantile')
    model_CI_lower.fit(X_train,y_train)

    return model_CI_upper, model_CI_lower

def get_bootstrap_mse_score_dist(estimator, X, y, num_bootstrap = 100, gridsearch=False):
    """Determines and plots the distribution of model performance (mse) using bootstrapping

    Parameters
    ----------
    estimator: sklearn-type class
        A model (class) that has been initialized with set parameters
    df: pandas DataFrame
        Full data set
    target : str
        Name of the target column in the dataset
    num_bootstrap : int
        Number of bootstrapping iterations to run

    Returns
    -------
    mse_scores : list of mse scores
    """

    if gridsearch:
        print "Running grid search ..."
        grid_params = {'learning_rate': [ 0.001, 0.01, 0.1],
                   'max_features': ['sqrt', 'log2', None],
                   'min_samples_leaf': [1,2,4],
                   'max_depth':[1,2,5],
                   'n_estimators': [500, 1000, 4000, 8000],
                   'subsample': [0.2, 0.5, 0.8]
                  }

        gdbr_gridsearch = GridSearchCV(estimator,
                                       grid_params, n_jobs=-1, verbose=True)

    mse_scores = []
    for i in range(num_bootstrap):
        if i%10==0:
            print "Running iteration {} ...".format(i)
            if i!=0:
                print "Mean of mse_scores = ", sum(mse_scores)/len(mse_scores)

        # Getting bootstrap indices for train
        boot_sample_range = np.array(range(0, len(X.index)))
        boot_sample_idx = np.random.choice(boot_sample_range, len(X.index), replace=True)
        boot_out_sample_idx = np.setdiff1d(boot_sample_range, boot_sample_idx)

        X_train = X.copy().iloc[boot_sample_idx]   # bootstrap sample
        y_train = y.copy().iloc[boot_sample_idx]

        # Assigning non-bootstrap indices for test set
        X_test = X.copy().iloc[boot_out_sample_idx]
        y_test = y.copy().iloc[boot_out_sample_idx]

        if gridsearch:

            gdbr_gridsearch.fit(X_train, y_train)
            estimator = gdbr_gridsearch.best_estimator_

        #print "Fitting model ..."
        estimator.fit(X_train, y_train)
        mse_scores.append(mean_squared_error(y_test, estimator.predict(X_test)))

    mse_scores_mean = sum(mse_scores)/len(mse_scores)

    print "Mean MSE:",mse_scores_mean
    return mse_scores

def plot_mse_distribution(mse_score_lists, color_list):
    """Fits GradientBoostingRegressor models for upper and lower bounds of confidence interval

    Parameters
    ----------
    mse_score_list : List of lists (mse scores from get_bootstrap_mse_score_dist())
    color_list : list of str (colors)

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    for i, mse_score_list in enumerate(mse_score_lists):
        ax.hist(mse_score_list, bins=30, alpha=0.3, color=color_list[i]);
        ax.axvline(sum(mse_score_list)/len(mse_score_list), color=color_list[i])
        print "{} MSE: {}".format(i, sum(mse_score_list)/len(mse_score_list))

    ax.yaxis.grid(True, linestyle='dotted', linewidth='0.5', color='gray')
    ax.set_facecolor("white")

def get_model_predictions_df(model_estimator, df, label, feature_list, dummy_list, index='Property_ID', conf_interval=0.90, loft_sample=False):
    """Fits GradientBoostingRegressor models for specified features and returns predictions

    Parameters
    ----------
    model_estimator: sklearn-type class
        A model (class) that has been initialized with set parameters
    df : pandas DataFrame
        Full data set
    label : str
        Name of the label column in the dataset that
    feature_list : list of str
        List of column names of the subset of features to be included in fitting the model
    dummy_list : list of str
        List of column names of the subset of features to be dummy-coded in fitting the model
    index: str, optional, default='Property_ID'
        Name of column to be used for pandas df joining process
    conf_interval: float, optional, default=0.90
        Confidence interval for predictions
    loft_sample: boolean, optional, default=False
        Set to true to use exact train-test split from loftium model

    Returns
    -------
    fitted_model : model class, fitted model on feature_list and dummy_list
    full_pred_df : pandas DataFrame, original df with predictions
    y_pred : array, predicted values
    X_train : pandas DataFrame, training data for inputs
    X_test : pandas DataFrame, testing data for inputs
    y_train : pandas DataFrame, training data for labels
    y_test : pandas DataFrame, test data for labels
    """

    print ("Running {} for label {}".format(model_estimator,label))
    dummy_df_list = get_dummy_dfs(df, dummy_list)
    X = prep_model_df(df, feature_list, dummy_df_list)
    X = X.drop(label, axis=1)
    y = df[label]

    # Perform train test split
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
    cv_mse = -get_cv_score(model_estimator, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_r2 = get_cv_score(model_estimator, X_train, y_train, cv=5, scoring='r2')
    print ("CV MSE: {}, CV R2: {}".format(cv_mse, cv_r2))
    print ("========")
    print ("Training model...")
    y_pred, fitted_model = get_model_pred(model_estimator, X_train, X_test,y_train)

    # Get confidence intervals for predictions
    fitted_model_CI_upper, fitted_model_CI_lower = get_model_confidence_interval(fitted_model, X_train, y_train, conf_interval)

    # Get error scores
    mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)
    print ("Test MSE: {}, Test R2: {}".format(mse, r2))
    print ("Returned pred df with shape {}".format(y_pred.shape))

    full_pred_df = df.copy()
    full_pred_df['pred'] = fitted_model.predict(X)
    full_pred_df['pred_upper'] = fitted_model_CI_upper.predict(X)
    full_pred_df['pred_lower'] = fitted_model_CI_lower.predict(X)

    return fitted_model, full_pred_df, y_pred, X_train, X_test, y_train, y_test

def plot_feature_importances(fitted_model, X_train):
    """Plots feature importance bar chart for fitted tree model

    Parameters
    ----------
    fitted_model: pre-fitted sklearn-type class
    X_train: pandas DataFrame

    Returns
    -------
    None
    """
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

def plot_partial_dependency_plots(fitted_model, X_train, n_row=3, n_col=3, xlabel_height=10, figsize=(12.0, 12.0)):
    """Plots n_row x n_col matrix of partial dependency plots for fitted tree model
    in descending order based on feature importance

    Parameters
    ----------
    fitted_model: pre-fitted sklearn-type class
    X_train: pandas DataFrame
    n_row: int
        Number of rows (optional, default = 3)
    n_col: int
        Number of columns (optional, default = 3)
    xlabel_height: int
        Position of label for feature importance value on plot (optional, default = 10)
    figsize: tuple
        Matplotlib figsize

    Returns
    -------
    None
    """

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
                             figsize=figsize)

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

def plot_cross_validation_train_and_test(model, X, y, N_FOLDS=5,N_ESTIMATORS = 2500):
    """A cross validation plotter that shows mse vs number of boosted stages

    Parameters
    ----------
    model : GradientBoostingRegressor model class
    X : array-like
        The training data to be used in fitting the model
    y : int, float, string or None, optional (default="auto")
        The labeled data
    N_FOLDS : integer or None, optional (default=10)
        The number of cross validation folds
    N_ESTIMATORS : integer or None, optional (default=4000)
        The number of decicion trees for boosting

    Returns
    -------
    None
    """

    train_scores = np.zeros((N_FOLDS, N_ESTIMATORS))
    test_scores = np.zeros((N_FOLDS, N_ESTIMATORS))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

    for k, (train_idxs, test_idxs) in enumerate(kf.split(X)):
        #print train_idxs
        X_train, y_train = X.iloc[train_idxs], y.iloc[train_idxs]
        X_test, y_test = X.iloc[test_idxs], y.iloc[test_idxs]
        model_cv = clone(model)
        #GradientBoostingRegressor(n_estimators=N_ESTIMATORS, learning_rate=0.01)
        model_cv.set_params(n_estimators=N_ESTIMATORS)
        model_cv.fit(X_train, y_train)
        for i, y_pred in enumerate(model_cv.staged_predict(X_train)):
            train_scores[k, i] = model_cv.loss_(y_train, y_pred)
        for i, y_pred in enumerate(model_cv.staged_predict(X_test)):
            test_scores[k, i] = model_cv.loss_(y_test, y_pred)

    mean_train_score = np.mean(train_scores, axis=0)
    mean_test_score = np.mean(test_scores, axis=0)

    optimal_n_trees = np.argmin(mean_test_score)
    optimal_score = mean_test_score[optimal_n_trees]
    optimal_point = (optimal_n_trees, optimal_score)

    for i in xrange(N_FOLDS):
        plt.plot(np.arange(N_ESTIMATORS) + 1, train_scores[i, :], color='red', alpha=0.25)

    for i in xrange(N_FOLDS):
        plt.plot(np.arange(N_ESTIMATORS) + 1, test_scores[i, :], color='blue', alpha=0.25)

    plt.plot(np.arange(N_ESTIMATORS) + 1, mean_test_score, color='blue', linewidth=2,
             label='Average Validation Fold Error')
    plt.plot(np.arange(N_ESTIMATORS) + 1, mean_train_score, color='red', linewidth=2,
             label='Average Training Fold Error')

    plt.annotate('Optimal CV Error', optimal_point,
                  xytext=(optimal_point[0] - 600, optimal_point[1] + 100),
                  arrowprops=dict(facecolor="darkgrey", shrink=0.05),
                  fontsize=14,
                  alpha=0.75
                )
    plt.title("Cross Validation Training and Testing Scores ({}) folds)".format(N_FOLDS))
    plt.xlabel('Number of Boosting Stages', fontsize=14)
    plt.ylabel('Average Squared Error', fontsize=14)
    #plt.yaxis.grid(True, linestyle='dotted', linewidth='0.5', color='gray')

    plt.legend(loc="upper right")
    print "Optimal point: {} trees, MSE {}".format(optimal_point[0], optimal_point[1])


def get_loftium_train_test_split(df):
    '''Conduct train-test split using loftium sample split'''

    insample_prop_ids = []
    with open('data/Loftium/insample_prop_ids.txt') as infile:
        for line in infile:
            insample_prop_ids.append(int(line.strip()))

    train_index = df.Property_ID.isin(insample_prop_ids)
    test_index = ~df.Property_ID.isin(insample_prop_ids)

    print (np.sum(train_index), np.sum(test_index))

    return train_index, test_index

def get_dummy_dfs(df, dummy_list):
    """Convert list of column names to dummy column dfs

    Parameters
    ----------
    df : pandas DataFrame
    dummy_list : dummy_list
        List of column names for columns to be converted to dummy columns

    Returns
    -------
    dummy_dfs : list
        List of pandas DataFrames with dummy-coded columns
    """

    dummy_dfs=[]
    for dummy in dummy_list:
        dummy_dfs.append(pd.get_dummies(df[dummy]))
    return dummy_dfs

def prep_model_df(df, features, dummy_dfs):
    """Convert list of column names to dummy column dfs

    Parameters
    ----------
    df : pandas DataFrame
    dummy_dfs : list
        List of pandas DataFrames with dummy-coded columns (from get_dummy_dfs())

    Returns
    -------
    model_df : pandas DataFrame
        DataFrame with appended dummy columns
    """

    model_df = df[features]
    for dummy_df in dummy_dfs:
        model_df = pd.concat([model_df,dummy_df],axis=1)
    return model_df
