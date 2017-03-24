"""
Module containing model fitting code for a web application that implements a
predictive model for Airbnb EDR.

python build_model.py
"""
#import statsmodels.api as sm
#import gmplot
#import seaborn as sns

import argparse
import cPickle as pickle
import pandas as pd
import numpy as np
import datetime
# NLP modules - to be incorporated later
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

import airbnb_EDA_helper as eda
import airbnb_prediction_helper as pred
from airbnKEY_model import AirbnKEY_Model, get_dummy_dfs, prep_model_df

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

def prep_monthly_data_for_model(excel_file, listing_data_df):
    """Clean and prepare imported data (monthly) for modeling

    Parameters
    ----------
    excel_file : str
        Path to excel file with data
    listing_data_df: pandas DataFrame
        Processed pandas Dataframe from function prep_annual_data_for_model()

    Returns
    -------
    listing_data_df : pandas DataFrame
        A pandas DataFrame with cleaned data
    """

    # Read in monthly data from excel file
    monthly_data_df = pd.read_excel(excel_file, sheetname='Monthly')
    monthly_data_df.columns = monthly_data_df.columns.str.replace(' ','_')

    # Filter monthly data to properties that are common with listing_data_df (annual data)
    prop_ids_to_keep = listing_data_df.Property_ID.unique()
    monthly_data_df = monthly_data_df[monthly_data_df.Property_ID.isin(prop_ids_to_keep)]

    # Cleaning up data, adding key features
    monthly_data_df['Property_Type'].replace('Bed &amp; Breakfast', 'Bed & Breakfast', inplace=True)
    monthly_data_df['Month'] = monthly_data_df.Reporting_Month.apply(lambda x: x.month)

    # Adding target column
    monthly_data_df['EDR'] = monthly_data_df.Occupancy_Rate * monthly_data_df.ADR

    # Only keeping months with more than 10 reservation days
    monthly_data_df = monthly_data_df[monthly_data_df.Reservation_Days>10]

    # Removing outlier property with minimal data and low review rating
    monthly_data_df = monthly_data_df[monthly_data_df.Property_ID!=7093910]

    return monthly_data_df

if __name__ == '__main__':

    # DATA LOAD AND PREPARATION
    # ==========================================================================
    print "Loading and preparing data ..."
    listing_data_df = prep_annual_data_for_model('../data/Loftium/Back testing - Copy.xlsx')
    monthly_data_df = prep_monthly_data_for_model('../data/Loftium/Back testing - Copy.xlsx', listing_data_df)

    # IMPORTANT: -------
    # The monthly data set contains only a subset of the feature space from the annual data.
    # To get access to full feature space at the monthly granularity, need to append features
    # from annual data to the monthly data.
    #   Step 1: Identify features from annual data to join with monthly data
    #   Step 2: Perform an inner join between the annual and monthly data on Property_ID

    # Step 1:  Identify features from annual data to join with monthly data
    listing_gdbr_cols_keep_merge = [
        'Property_ID',   # Need to add this for joining
        'Annual_Revenue_LTM',
        'Average_Daily_Rate',
        'Bathrooms',
        'Calendar_Last_Updated',
        'Cancellation_Policy',
        'Checkin_Time',
        'Checkout_Time',
        'Cleaning_Fee',
        'Count_Available_Days_LTM',
        'Count_Blocked_Days_LTM',
        'Count_Reservation_Days_LTM',
        'Created_Date',
        'Days_Since_Created',
        'Extra_People_Fee',
        'Instantbook_Enabled',
        'Last_Scraped_Date',
        'Listing_Main_Image_URL',
        'Listing_Title',
        'Listing_URL',
        'Max_Guests',
        'Minimum_Stay',
        'Number_of_Bookings_LTM',
        'Number_of_Photos',
        'Number_of_Reviews',
        'Occupancy_Calculated',
        'Occupancy_Rate_LTM',
        'Overall_Rating',
        'Pike_Market',
        'Published_Monthly_Rate',
        'Published_Nightly_Rate',
        'Published_Weekly_Rate',
        'Response_Rate',
        'Response_Time_min',
        'Security_Deposit',
        'Superhost',
        'Zipcode',
        'nonroom',
        'parking',
        'private_bath',
        'view',
        'walk',
        'water',
    ]

    # Step 2: Perform an inner join between the annual and monthly data on Property_ID
    print "Joining monthly and annual datasets ..."
    merge_df = pd.merge(monthly_data_df,
                        listing_data_df[listing_gdbr_cols_keep_merge],
                        left_on='Property_ID',  right_on='Property_ID',
                        how='inner')

    # FEATURE ENGINEERING
    # ==========================================================================
    # Now that we have access to the the full feature space at the monthly granularity,
    # down-select to features to use in predictive model.
    #
    # The goal is to only keep features that will have meaningful contributions to predictive accuracy.

    # Select continuous features to be used in model training
    merge_gdbr1_cols_keep = [
        # ==== LISTING DETAILS ==========a
        #'Bathrooms', 'Bedrooms',
        'Days_Since_Created', 'Instantbook_Enabled',
        #'Max_Guests', 'Minimum_Stay', 'Number_of_Photos',
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
        #'Security_Deposit', 'Extra_People_Fee', 'Cleaning_Fee',

        # ==== HOST QUALITY ============
        'Superhost', 'Overall_Rating','Number_of_Reviews',
        # ---- Exclude
        #'Response_Rate', 'Response_Time_min',

        # ==== SPECIAL FEATURES ========
        'private_bath', 'view',
        # ---- Exclude
        #'Pike_Market', 'nonroom',
        #'water','parking',
        #'walk',
    ]

    # Select categoriccal featues to be converted into dummy columns
    merge_gdbr1_cols_dummy =[
        #'Neighborhood',   # Removing since location information captured by lat-long
        #'Zipcode',        # Zipcode is from annual dataset
        #'Zip_code',       # Zip_code is from monthly dataset
        'Month',
        'Property_Type',
    ]

    print "Reducing feature space ..."
    # Reduce feature space and add columns with dummy coding as specified above
    merge_gdbr1_dummy_dfs = get_dummy_dfs(merge_df, merge_gdbr1_cols_dummy)
    X = prep_model_df(merge_df, merge_gdbr1_cols_keep, merge_gdbr1_dummy_dfs)

    # Specify the target variable
    #   EDR = Expected Daily Rate for a given month
    y = X.pop('EDR')

    # Performing train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    # Setting the model
    model = AirbnKEY_Model(GradientBoostingRegressor(learning_rate=0.05, n_estimators=2500),
                               conf_interval=0.90)
    print "Training model ..."
    # Fitting the model
    model.fit(X_train, y_train)

    # =========================================================================
    # ANNUAL model
    # This section is for model creation at an annual granularity.
    # Currently not being used, since seasonality is not captured at annual level.
    # =========================================================================
    # listing_gdbr_cols_keep = [
    #     # ==== LISTING DETAILS ==========
    #     #'Bathrooms', 'Bedrooms',
    #     'Days_Since_Created', 'Number_of_Photos','Instantbook_Enabled',
    #     'Max_Guests', 'Minimum_Stay',
    #     # ---- Exclude
    #     #'Property_ID', 'Listing_Title', 'Listing_Main_Image_URL','Listing_URL','Created_Date',
    #     #'Check-in_Time', 'Checkout_Time',
    #
    #     # ==== BOOKING HISTORY =========
    #     'EDR',
    #     # ---- Exclude
    #     #'Occupancy_Calculated', 'Annual_Revenue_LTM', 'Average_Daily_Rate' 'Last_Scraped_Date','Calendar_Last_Updated',
    #     #'Number_of_Bookings_LTM','Count_Blocked_Days_LTM', 'Count_Reservation_Days_LTM','Count_Available_Days_LTM',
    #
    #     # ==== LOCATION ================
    #     'Latitude', 'Longitude',
    #     # ---- Exclude
    #     #'Country', 'State','City', 'Metropolitan_Statistical_Area',
    #
    #     # ==== EXTENDED STAY DETAILS ===
    #     # --- -Exclude
    #     #'Published_Monthly_Rate','Published_Nightly_Rate', 'Published_Weekly_Rate',
    #
    #     # ==== ADDITIONAL FEES =========
    #     'Security_Deposit', 'Extra_People_Fee', 'Cleaning_Fee',
    #
    #     # ==== HOST QUALITY ============
    #     'Superhost', 'Overall_Rating','Number_of_Reviews',
    #     # ---- Exclude
    #     #'Response_Rate', 'Response_Time_min',
    #
    #     # ==== SPECIAL FEATURES ========
    #     'Pike_Market', 'nonroom', 'private_bath', 'view', 'water','parking',
    #     # ---- Exclude
    #     #'walk',
    #     ]

    # Features to dummy
    # listing_gdbr_cols_dummy = [
    #     #'Neighborhood',
    #     #'Zipcode',
    #     'Cancellation_Policy',
    #     'Property_Type'
    #     ]

    # From gridsearch, following hyperparameters:
    # {'learning_rate': 0.01,
    # 'max_depth': 5,
    # 'max_features': 'sqrt',
    # 'min_samples_leaf': 2,
    # 'n_estimators': 1000}

    # Defining the best model
    # best_model = GradientBoostingRegressor(  alpha=0.9, criterion='friedman_mse', init=None,
    #                                          learning_rate=0.01, loss='ls', max_depth=5,
    #                                          max_features='sqrt', max_leaf_nodes=None,
    #                                          min_impurity_split=1e-07, min_samples_leaf=2,
    #                                          min_samples_split=2, min_weight_fraction_leaf=0.0,
    #                                          n_estimators=1000, presort='auto', random_state=None,
    #                                          subsample=1.0, verbose=0, warm_start=False)
    # =========================================================================
    # ANNUAL model - end
    # =========================================================================


    # Saving trained model to a pickle object for use in flask application
    # =========================================================================
    datestring = '{:04d}{:02d}{:02d}{:02d}{:02d}'.format(datetime.date.today().year,
                                                         datetime.date.today().month,
                                                         datetime.date.today().day,
                                                         datetime.datetime.today().hour,
                                                         datetime.datetime.today().minute,)

    filename ='../app/static/gdbr_model.pkl'
    filename_backup ='../app/static/gdbr_model_{}.pkl'.format(datestring)

    print ("Saving model to {}".format(filename))

    with open(filename_backup, 'w') as f:
        pickle.dump(model, f)

    with open(filename, 'w') as f:
        pickle.dump(model, f)
