import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.grid_search import GridSearchCV
import statsmodels.api as sm
from scipy.stats import f_oneway
#from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib
import shapefile

#import seaborn as sns
from seaborn import  diverging_palette, heatmap

def check_for_null(df):
    null_columns=[]
    for column in df.columns:
        null_count = df[column].isnull().sum()
        if null_count >0:
            print ("***WARNING*** : {} null values in {}".format(null_count, column))
            null_columns.append(column)

    return null_columns

def plot_neighborhoods(xmin, xmax, ymin, ymax, shapes_file, figsize):

    ctr = shapefile.Reader(shapes_file)
    geomet = ctr.shapeRecords() #will store the geometry separately

    fig, ax = plt.subplots(1,figsize=figsize)

    patches=[]
    for point in geomet:
        nhood = point.record[3]
        polygon_pts=[]
        ctr_pt =np.mean(np.array(point.shape.points), axis=0)
        if ((ctr_pt[0] > xmin) & (ctr_pt[0] < xmax)) & ((ctr_pt[1] > ymin) & (ctr_pt[1] < ymax)):
            for x, y in point.shape.points:
                pts = (x,y)
                polygon_pts.append(pts)
            patch = Polygon(polygon_pts, label=nhood)
            ax.text(ctr_pt[0], ctr_pt[1], nhood)
            patches.append(patch)

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.3, edgecolors='black')
    colors = 100*np.random.rand(len(patches))
    p.set_array(np.array(colors))

    ax.add_collection(p)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    #ax.set_facecolor(None)
    ax.set_alpha(0.3)
    #ax.axis('off')
    plt.grid(True, linestyle='dashed', color='#E3E3E3')
    plt.show()

# This is a color map that ensures that each group on a plot has a different color
def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def plot_lat_long_sizes(df, column, y='price', figsize=(10,8), size_scale=1):
    fig, ax = plt.subplots(1,1, figsize=figsize)

    unique_list = df[column].unique()
    N = len(unique_list)
    cmap = get_cmap(N)

    for i,item in enumerate(unique_list):
        col = cmap(i)
        ax.scatter(df[df[column]==item].longitude,
                   df[df[column]==item].latitude,
                   alpha=0.3,
                   s=df[df[column]==item][y]*size_scale,
                   c=col,
                   label=item)
        ax.set_ylim(47.50, 47.75)
        ax.set_xlim(-122.45, -122.20)
        ax.legend(fontsize='small')
        title_s = "Listings by {} (Size = {})".format(column, y)
        ax.set_title(title_s)

    plt.show()


def plot_boxplot_sorted(df, by, column, rot=0, fontsize=8, figsize=(12,6), jitter_offset=0.0, sort_flag=True, show_outliers=True):

    fig, ax = plt.subplots(1,figsize=figsize)
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    meds = df2.median().sort_values(ascending=False)
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    if sort_flag:
        for i,d in enumerate(df2[meds.index]):
            y = df2[meds.index][d]
            x = np.random.normal(i+1+jitter_offset, 0.04, len(y))
            ax.plot(x, y, ms=3, marker="o", linestyle="None", alpha=0.3, c='blue')
        #ax.boxplot()
        df2[meds.index].boxplot(rot=rot, return_type="axes",  fontsize=fontsize, showmeans=True, widths=0.7)

    else:
        for i,d in enumerate(df2):
            y = df2[d]
            x = np.random.normal(i+1+jitter_offset, 0.04, len(y))
            ax.plot(x, y, ms=3, marker="o", linestyle="None", alpha=0.3, c='blue')
        #ax.boxplot()
        df2.boxplot(rot=rot, return_type="axes",  fontsize=fontsize, showmeans=True, widths=0.7)

    ax.set_ylabel=(column)
    ax.set_title("{} by {}\n".format(column, by))

    if not show_outliers:
        iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
        upper_lim = df[column].quantile(0.75)+3*iqr
        ax.set_ylim(0,upper_lim)
    plt.show()

def plot_boxplot_compare(df, by, column1, column2, rot=0, fontsize=8, figsize=(12,6), jitter_offset=0.0, sort_flag=True, show_outliers=True):

    fig, ax = plt.subplots(1,figsize=figsize)

    sorted_groups = df.groupby(by).mean()[column1].sort_values(ascending=False).index

    capprops1 = dict(linestyle='')
    whiskerprops1 = dict(linestyle='')
    medianprops1 = dict(linestyle='--', linewidth=2., color='black')
    meanprops1 = dict(linestyle='-', linewidth=2., color='black')
    boxprops1 = dict(linestyle='')
    data1 = [df[df[by]==group][column1] for group in sorted_groups ]
    box1 = ax.boxplot(data1, labels=sorted_groups, widths=0.4, showmeans=True, meanline=True,
                      boxprops=boxprops1, meanprops=meanprops1, medianprops=medianprops1, capprops=capprops1, whiskerprops=whiskerprops1);

    capprops2 = dict(linestyle='--', linewidth=1.0, color='green')
    whiskerprops2 = dict(linestyle='--', linewidth=0.5, color='green')
    boxprops2 = dict(linestyle='--', linewidth=0.5, color='green')
    meanprops2 = dict(linestyle='-', linewidth=2., color='green')
    medianprops2 = dict(linestyle='--', linewidth=2., color='green')
    data2 = [df[df[by]==group][column2] for group in sorted_groups ]
    box2 = ax.boxplot(data2, labels=sorted_groups, patch_artist=True, widths=0.4, showmeans=True,meanline=True,
                      boxprops=boxprops2, meanprops=meanprops2, medianprops=medianprops2, capprops=capprops2, whiskerprops=whiskerprops2)

    for box in box2['boxes']:
        box.set_alpha(0.3)
        box.set_facecolor('green')

    for i,d in enumerate(sorted_groups):
        y = df[df[by]==d][column1]
        x = np.random.normal(i+1+jitter_offset, 0.02, len(y))
        ax.plot(x, y, ms=3, marker="o", linestyle="None", alpha=0.5, c='black')
    #ax.boxplot()
    for i,d in enumerate(sorted_groups):
        y = df[df[by]==d][column2]
        x = np.random.normal(i+1.1+jitter_offset, 0.02, len(y))
        ax.plot(x, y, ms=3, marker="o", linestyle="None", alpha=0.5, c='green')
    #ax.boxplot()

    ax.set_ylabel(column1)
    ax.set_title("{} by {}\n".format(column1, by), fontsize=fontsize)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=rot)
    ax.set_facecolor('white')
    #x.grid(linestyle='dotted', linewidth='0.5', color='gray')
    ax.yaxis.grid(True, linestyle='dotted', linewidth='0.5', color='gray')
    #ax.set_ylabel('EDR by month')
    # if not show_outliers:
    #     iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    #     upper_lim = df[column].quantile(0.75)+3*iqr
    #     ax.set_ylim(0,upper_lim)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if not show_outliers:
        iqr = df[column1].quantile(0.75) - df[column1].quantile(0.25)
        upper_lim = df[column1].quantile(0.75)+3*iqr
        ax.set_ylim(0,upper_lim)

    plt.show()


def plot_corr_matrix_heatmap(df, annot=True):
    corr_matrix = df.corr(method='pearson', min_periods=1)
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.8,
            square=True, annot=annot,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.show()

def plot_prediction_on_data(x, y_pred, y_true):
    plt.plot(x,y_pred, color='red')
    plt.scatter(x,y_true, color='blue', alpha=0.3)
    plt.show()


def plot_residuals(x, resid):
    plt.scatter(x,resid, color='blue', alpha=0.5)
    plt.plot((-5, 40), (0, 0), '--', color='black')
    plt.xlabel("X")
    plt.ylabel("Residuals")
    plt.show()

def get_linreg_summary(x,y, xcolumns, standardization=False):
    if standardization:
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('lr', LinearRegression()))
        linreg = Pipeline(estimators)
        linreg.fit(x,y)
        coefs = linreg.steps[1][1].coef_
        intercept = linreg.steps[1][1].intercept_
    else:
        linreg = LinearRegression()
        linreg.fit(x,y)
        coefs = linreg.coef_
        intercept = linreg.intercept_

    print ("  Intercept: {:7.3f}".format( intercept))
    for coef, feature in zip(coefs, xcolumns):
        print ("{:>10} : {:7.3f}".format(feature, coef))

    print (" Score R^2 : {:7.3f}".format(linreg.score(x,y)))
    print (" Score MSE : {:7.3f}".format(mean_squared_error(y,linreg.predict(x))))
    return linreg

def get_linreg_summary_sm(model):

    print (model.summary())

    fig, ax = plt.subplots(1,1,figsize=(8,4))
    ax.scatter(model.fittedvalues, model.outlier_test()['student_resid'],
                alpha=0.3)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Studentized residuals')
    plt.show()

def score_model(estimator, X, y, kfolds=10):
    mse = -np.mean(cross_val_score(estimator, X_train, y_train,
                                   cv=kfolds, scoring='neg_mean_squared_error'))
    r2 = np.mean(cross_val_score(estimator, X_train, y_train,
                                 cv=kfolds, scoring='r2'))

    return "{:>30} Train CV | MSE: {:7.3f} | R2: {:7.3f}".format(estimator.__class__.__name__,
                                                                mse,
                                                                r2)

def plot_grad_boost_stage_scores(grad_boost_model, X_train, y_train, X_test, y_test):

    # First, fit the model on training data
    train_scores = np.zeros(model.n_estimators)
    test_scores = np.zeros(model.n_estimators)

    for i, y_train_predict in enumerate(grad_boost_model.staged_predict(X_train)):
        train_scores[i] = mean_squared_error(y_train, y_train_predict)

    for i, y_test_predict in enumerate(grad_boost_model.staged_predict(X_test)):
        test_scores[i] = mean_squared_error(y_test, y_test_predict)

    label_train = "{} Train - Learning Rate {}".format(grad_boost_model.__class__.__name__,
                                                       grad_boost_model.learning_rate)
    label_test = "{} Test - Learning Rate {}".format(grad_boost_model.__class__.__name__,
                                                       grad_boost_model.learning_rate)

    linreg_mse = -np.mean(cross_val_score(LinearRegression(), X_train, y_train,
                          cv=10, scoring='neg_mean_squared_error'))

    rf_mse = -np.mean(cross_val_score(RandomForestRegressor(), X_train, y_train,
                      cv=10, scoring='neg_mean_squared_error'))

    plt.plot(train_scores, label=label_train)
    #plt.plot(test_scores, label=label_test)
    plt.axhline(linreg_mse, linewidth=1, color='black', linestyle='--', label="Linear Regression CV")
    plt.axhline(rf_mse, linewidth=1, color='red', linestyle='--', label="Random Forest CV")
    plt.legend(fontsize=8)
    plt.xlabel("Number of Boosted Stages")
    plt.ylabel("MSE")
    plt.show()

def impute_null_columns(df):
    # Converting flags to 1/0
    df_new = df.copy()
    df_new.Instantbook_Enabled.replace({'Yes':1, 'No':0}, inplace=True)
    df_new.Superhost.replace({'t':1, 'f':0}, inplace=True)

    # Cleaning up null values
    df_new.Superhost.fillna(0, inplace=True)
    df_new.Security_Deposit.fillna(0, inplace=True)
    df_new.Cleaning_Fee.fillna(0, inplace=True)
    df_new.Extra_People_Fee.fillna(0, inplace=True)
    df_new.Bathrooms.fillna(0, inplace=True)

    df_new.Overall_Rating.fillna(df_new.Overall_Rating.mean(), inplace=True)
    df_new.Published_Monthly_Rate.fillna(df_new.Published_Monthly_Rate.mean(), inplace=True)
    df_new.Published_Weekly_Rate.fillna(df_new.Published_Weekly_Rate.mean(), inplace=True)
    df_new.Calendar_Last_Updated.fillna(datetime.datetime(2016,1,1), inplace=True)
    df_new.Response_Rate.fillna(df_new.Response_Rate.mean(), inplace=True)
    df_new.Response_Time_min.fillna(df_new.Response_Time_min.mean(), inplace=True)

    df_new.Neighborhood.fillna('NA', inplace=True)
    df_new.Neighborhood.fillna('NA', inplace=True)
    df_new.Listing_Title.fillna('NA', inplace=True)
    df_new.Host_ID.fillna('NA', inplace=True)
    df_new.Checkin_Time.fillna('NA', inplace=True)
    df_new.Checkout_Time.fillna('NA', inplace=True)

    return df_new
