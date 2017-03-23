"""
Module containing model fitting code for a web application that implements a
text classification model.

When run as a module, this will load a csv dataset, train a classification
model, and then pickle the resulting model object to disk.

USE:

python build_model.py --data path_to_input_data --out path_to_save_pickled_model
python build_model.py --data data/articles.csv --out static/model.pkl
"""
#import statsmodels.api as sm
#from scipy.misc import imread
#import gmplot
#import seaborn as sns

import argparse
import cPickle as pickle
import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

import airbnb_EDA_helper as eda
import airbnb_prediction_helper as pred

class Airbnb_EDR_Model(object):

    def __init__(self, estimator):
        self.estimator = estimator
        self.X_train = None
        self.y_train = None
        self.cv_score_ = None
        self.test_mse_ = None


    def fit(self, X_train, y_train):
        self.X_train=X_train
        self.y_train=y_train
        self.estimator.fit(X_train, y_train)
        return self.estimator

    def cv_mse(self,cv=5):
        self.cv_mse_ = np.mean(cross_val_score(self.estimator, self.X_train, self.y_train,
                                                 cv=cv, scoring='neg_mean_squared_error'))
        return -self.cv_mse_

    def predict(self,X):
        return self.estimator.predict(X)

    def test_mse(self,X,y):
        y_pred = self.predict(X)
        self.test_mse_ = mean_squared_error(y_pred, y)
        return self.test_mse_

def get_dummy_dfs(df, dummy_list):
    """Append dummy columns to dataframe

    Parameters
    ----------
    df : pandas DataFrame
        Complete dataset to be used for model training
    dummy_list : list
        List of column names to be converted to dummy columns

    Returns
    -------
    dummy_list : list
        A list of pandas DataFrames of dummy columns
    """

    dummy_dfs=[]
    for dummy in dummy_list:
        dummy_dfs.append(pd.get_dummies(df[dummy]))
    return dummy_dfs

def prep_model_df(df, features, dummys):
    """Append dummy columns to dataframe

    Parameters
    ----------
    df : pandas DataFrame
        Complete dataset to be used for model training
    dummy_list : list
        List of pandas dfs of dummy coded columns

    Returns
    -------
    model_df : pandas DataFrame
        An expanded pandas DataFrame with appended dummy columns
    """
    model_df = df[features]
    for dummy in dummys:
        model_df = pd.concat([model_df,dummy],axis=1)
    return model_df


def prep_annual_data_for_model(excel_file):
    """Clean and prepare imported data (annual) for modeling

    Parameters
    ----------
    excel_file : str
        Path to excel file with data

    Returns
    -------
    listing_data_df : pandas DataFrame
        A pandas DataFrame with cleaned data
    """
    # Load data
    listing_data_df = pd.read_excel(excel_file, sheetname='Listings')
    listing_data_df.columns = listing_data_df.columns.str.replace(' ','_').str.replace('\(','').str.replace('\)','').str.replace('-','')
    listing_data_df.Listing_Title = listing_data_df.Listing_Title.str.lower()

    # Cleaning up nulls and encoding issues
    listing_data_df = eda.impute_null_columns(listing_data_df)
    listing_data_df['Property_Type'].replace('Bed &amp; Breakfast', 'Bed & Breakfast', inplace=True)

    # Adding flags for views, water, priv bathrooms, parking
    listing_data_df['view'] = listing_data_df.Listing_Title.str.contains(' view').astype(int)
    water_pattern = r'(( water|^water|saltwater| lake|^lake|greenlake| bay|^bay).*(view))|(view).*( water|^water|saltwater| lake|^lake|greenlake| bay|^bay)'
    listing_data_df['water'] = listing_data_df.Listing_Title.str.contains(water_pattern).astype(int)
    listing_data_df['private_bath'] = listing_data_df.Listing_Title.str.contains('bath').astype(int)
    listing_data_df['walk'] = (listing_data_df.Listing_Title.str.contains('walk')).astype(int)
    listing_data_df['parking'] = (listing_data_df.Listing_Title.str.contains('parking')).astype(int)
    listing_data_df['Property_Type'].replace('Bed &amp; Breakfast', 'Bed & Breakfast', inplace=True)
    # Adding flags for nonrooms
    nonroom_pattern = r'((couch|futon|space))'
    listing_data_df['nonroom'] = ((listing_data_df.Listing_Title.str.contains(nonroom_pattern)) & \
                                  ~(listing_data_df.Listing_Title.str.contains('needle'))).astype(int)
    # Adding flag for Pike Place Market
    listing_data_df['Pike_Market'] = (listing_data_df.Neighborhood=='Pike-Market').astype(int)

    # Converting Created_Date to Days_Since_Created
    listing_data_df['Days_Since_Created'] = listing_data_df.Created_Date.apply(lambda x: (listing_data_df.Created_Date.max()-x).days)
    listing_data_df['EDR'] = listing_data_df.Occupancy_Calculated * listing_data_df.Average_Daily_Rate

    # Only keeping relevant properties
    listing_data_df = listing_data_df[listing_data_df.Number_of_Bookings_LTM > 4]
    listing_data_df = listing_data_df[listing_data_df.Occupancy_Calculated > .5]
    features_Property_type_keep = ['House', 'Apartment','Townhouse','Condominium', 'Bed & Breakfast','Loft','Other']
    listing_data_df = listing_data_df[listing_data_df.Property_Type.isin(features_Property_type_keep)]

    return listing_data_df

if __name__ == '__main__':

    listing_data_df = prep_annual_data_for_model('../data/Loftium/Back testing - Copy.xlsx')
    # Feature engineering
    listing_gdbr_cols_keep = [
        # ==== LISTING DETAILS ==========
        #'Bathrooms', 'Bedrooms',
        'Days_Since_Created', 'Number_of_Photos','Instantbook_Enabled',
        'Max_Guests', 'Minimum_Stay',
        # ---- Exclude
        #'Property_ID', 'Listing_Title', 'Listing_Main_Image_URL','Listing_URL','Created_Date',
        #'Check-in_Time', 'Checkout_Time',

        # ==== BOOKING HISTORY =========
        'EDR',
        # ---- Exclude
        #'Occupancy_Calculated', 'Annual_Revenue_LTM', 'Average_Daily_Rate' 'Last_Scraped_Date','Calendar_Last_Updated',
        #'Number_of_Bookings_LTM','Count_Blocked_Days_LTM', 'Count_Reservation_Days_LTM','Count_Available_Days_LTM',

        # ==== LOCATION ================
        'Latitude', 'Longitude',
        # ---- Exclude
        #'Country', 'State','City', 'Metropolitan_Statistical_Area',

        # ==== EXTENDED STAY DETAILS ===
        # --- -Exclude
        #'Published_Monthly_Rate','Published_Nightly_Rate', 'Published_Weekly_Rate',

        # ==== ADDITIONAL FEES =========
        'Security_Deposit', 'Extra_People_Fee', 'Cleaning_Fee',

        # ==== HOST QUALITY ============
        'Superhost', 'Overall_Rating','Number_of_Reviews',
        # ---- Exclude
        #'Response_Rate', 'Response_Time_min',

        # ==== SPECIAL FEATURES ========
        'Pike_Market', 'nonroom', 'private_bath', 'view', 'water','parking',
        # ---- Exclude
        #'walk',
        ]

    # Features to dummy
    listing_gdbr_cols_dummy = [
        #'Neighborhood',
        #'Zipcode',
        'Cancellation_Policy',
        'Property_Type'
        ]

    # From gridsearch, following hyperparameters:
    # {'learning_rate': 0.01,
    # 'max_depth': 5,
    # 'max_features': 'sqrt',
    # 'min_samples_leaf': 2,
    # 'n_estimators': 1000}

    # Defining the best model
    best_model = GradientBoostingRegressor(  alpha=0.9, criterion='friedman_mse', init=None,
                                             learning_rate=0.01, loss='ls', max_depth=5,
                                             max_features='sqrt', max_leaf_nodes=None,
                                             min_impurity_split=1e-07, min_samples_leaf=2,
                                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                                             n_estimators=1000, presort='auto', random_state=None,
                                             subsample=1.0, verbose=0, warm_start=False)

    # Prepare pandas dataframe for modeling:
    #   1) Create dataframes with dummy columns
    #   2) Combine dummy columns with main dataframe
    dummy_df_list = pred.get_dummy_dfs(listing_data_df,listing_gdbr_cols_dummy)
    model_df = pred.prep_model_df(listing_data_df,listing_gdbr_cols_keep,dummy_df_list)

    X = model_df.drop('EDR', axis=1)
    y = model_df.EDR

    model = best_model.fit(X, y)

    datestring = '{:04d}{:02d}{:02d}'.format(datetime.date.today().year,
                                         datetime.date.today().month,
                                         datetime.date.today().day,
                                         datetime.datetime.today().hour,
                                         datetime.datetime.today().minute,)

    filename ='../app/static/model_{}.pkl'.format(datestring)

    print ("Saving model to {}".format(filename))

    with open(filename, 'w') as f:
        pickle.dump(model, f)
