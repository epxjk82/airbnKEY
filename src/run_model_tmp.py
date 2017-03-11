
# coding: utf-8

# # Fraud case study
# 
# 

# ## Day 1: building a fraud model

# ## Day 2: building an app/dashboard

# ## Tips success
# 
# You will quickly run out of time:
# 
# *  Use CRISP-DM workflow to analyze data and build a model
# *  Iterate quickly, test often, commit often
# *  Build deadlines for your work so you stay on track
# *  Should have a model by end of day 1
# *  Start app once model is working

# ### CRISP-DM workflow
# 
# Follow the [CRISP-DM](https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining) steps:
# 
# 1.  Business understanding
# 2.  Data understanding
# 3.  Data preparation
# 4.  Modeling
# 5.  Evaluation
# 6.  Deployment

# # Introduction to case study: data & problem

# Let's look at the data.  What format is the data in?  How do you extract it?

# In[1]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[110]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

import src.data_cleanup as dc
import src.model_helper as mh


# In[3]:

ls -lh data


# Unzip the data so you can load it into Python

# In[4]:

#!unzip data/data.zip -d data


# Initially, work with a subset at first in order to iterate quickly.  But, the file is one giant line of json:

# In[5]:

get_ipython().system(u'wc data/data.json')


# Write a quick and dirty script to pull out the first 100 records so we can get code working quickly.

# In[61]:

get_ipython().run_cell_magic(u'writefile', u'subset_json.py', u'"""head_json.py - extract a couple records from a huge json file.\n\nSyntax: python head_json.py < infile.json > outfile.json\n"""\n\nimport sys\n\nstart_char = \'{\'\nstop_char = \'}\'\nn_records = 2000\nlevel_nesting = 0\n\nwhile n_records != 0:\n    ch = sys.stdin.read(1)\n    sys.stdout.write(ch)\n    if ch == start_char:\n        level_nesting += 1\n    if ch == stop_char:\n        level_nesting -= 1\n        if level_nesting == 0:\n            n_records -= 1\nsys.stdout.write(\']\')')


# In[62]:

get_ipython().system(u'python subset_json.py < data/data.json > data/subset.json')


# In[63]:

df = pd.read_json('data/subset.json')


# # Data cleanup

# In[64]:

df.info()


# In[65]:

numeric_col = df.select_dtypes(include=[np.number]).columns
categorical_col = np.setdiff1d(df.columns, numeric_col)


# In[66]:

col_convert_to_dates=[
'approx_payout_date',
'event_created',
'event_end',
'event_published',
'event_start'
'user_created'
]


# In[67]:

col_get_dummies = [
# Numerical
'delivery_method',  
'user_type',
'org_facebook',
'org_twitter', 
# Categorical
'acct_type',
'country',
'currency',
'description',
'email_domain',
'listed',
'name',
'org_desc',
'org_name',
'payee_name',
'payout_type',
'previous_payouts',
'ticket_types',
'venue_address',
'venue_country',
'venue_name',
'venue_state'
]


# In[68]:

cols_null_zero = [
'body_length',
'channels',
'fb_published',  
'gts',
'has_analytics',
'has_header',
'has_logo',
'name_length',
'num_order',
'num_payouts',
#'object_id',
'sale_duration',
'sale_duration2',
'show_map',
'user_age',
# Consider removing these later?  Use lasso to evaluate. 
]


# In[69]:

cols_null_mean = [
'venue_latitude',
'venue_longitude',      
'approx_payout_date',  
'event_created',
'event_end',
'event_published',
'event_start',
'user_created'
]


# In[70]:

cols_null_dummy = [
# Numerical
'delivery_method',  
'user_type',
'org_facebook',
'org_twitter', 
# Categorical
'acct_type',
'country',
'currency',
'description',
'email_domain',
'listed',
'name',
'event_published',
'org_desc',
'org_name',
'payee_name',
'payout_type',
'previous_payouts',
'ticket_types',
'venue_state',
'venue_address',
'venue_country',
'venue_name',
]


# In[71]:

clean_df = dc.clean_up_nulls(df, 
                             cols_null_zero,
                             cols_null_mean,
                             cols_null_dummy)


# In[72]:

#Check for nulls
for col in clean_df.columns:
    null_cnt = sum(clean_df[col].isnull())
    if null_cnt > 0:
        print col, null_cnt


# # First Model

# ### Let's try logistic regression first
# ### Must remove non-numerical data

# # Create label column

# In[73]:

# 1 if fraud, 0 if legit
fraud_list = ['fraudster', 'fraudster_event']
fraud = (df.acct_type.isin(fraud_list)).astype(int)


# Some of the data is text (and HTML), which will require feature engineering:
# 
# * TF-IDF
# * Feature hashing
# * n-grams
# 
# etc.
# 
# You will also need to construct a target from `acct_type`.  Fraud events start with `fraud`.  How you define fraud depends on how you define the business problem.

# Is missing data a problem?  What are your options for handling missing data?

# In[74]:

#clean_df.head().T


# In[75]:

numeric_col_clean = clean_df.select_dtypes(include=[np.number]).columns
categorical_col_clean = np.setdiff1d(clean_df.columns, numeric_col_clean)


# In[76]:

numeric_col_clean_mod = [  
#'approx_payout_date',
'body_length',
'channels',
#'delivery_method',
'event_created',
'event_end',
'event_published',
'event_start',
'fb_published',
'gts',
'has_analytics',
'has_header',
'has_logo',
'name_length',
'num_order',
'num_payouts',
#'object_id',
#'org_facebook',
#'org_twitter',
'sale_duration',
'sale_duration2',
'show_map',
'user_age',
'user_created',
'user_type',
'venue_latitude',
'venue_longitude',
'delivery_method_null',
'user_type_null',
'org_facebook_null',
'org_twitter_null',
'acct_type_null',
'country_null',
'currency_null',
'description_null',
'email_domain_null',
'listed_null',
'name_null',
'event_published_null',
'org_desc_null',
'org_name_null',
'payee_name_null',
'payout_type_null',
'previous_payouts_null',
'ticket_types_null',
'venue_state_null',
# 'venue_address_null',
# 'venue_country_null',
'venue_name_null',   
]


# In[122]:

categorical_col_clean_mod = [
'acct_type',
'country',
'currency',
'description',
'email_domain',
'listed',
'name',
'org_desc',
'org_name',
'payee_name',
'payout_type',
'previous_payouts',
'ticket_types',
'venue_address',
'venue_country',
'venue_name',
'venue_state',
]


# In[78]:

# for col in clean_df.columns:
#     print "'{}',".format(col)


# ## Let's feature engineer!

# In[143]:

# If the count of previous payouts is zero, probably a high risk.
# Let's convert previous_payouts to previous_payouts_cnt (count)
clean_df['previous_payouts_cnt'] = clean_df.previous_payouts.apply(lambda x: len(x))


# In[145]:

df.columns


# In[144]:

df['email_domain',''


# In[142]:

addl_col = [
'previous_payouts_cnt'      
]


# In[133]:

model_df = clean_df[numeric_col_clean_mod + addl_col]


# In[134]:

X = model_df
y = fraud


# In[135]:

X_train, X_test, y_train, y_test = train_test_split(X,y)


# ```
# Confusion Matrix
# # [[tn, fp]]
# # [[fn, tp]]
# ```

# In[136]:

mh.get_model_confusion_matrix(LogisticRegression, X_train, y_train)


# In[137]:

mh.get_model_confusion_matrix(RandomForestClassifier, X_train, y_train)


# In[138]:

mh.get_model_confusion_matrix(GradientBoostingClassifier, X_train, y_train)


# In[139]:

mh.get_cross_val_accuracy(LogisticRegression, X_train, y_train)


# In[140]:

mh.get_cross_val_accuracy(RandomForestClassifier, X_train, y_train)


# In[141]:

mh.get_cross_val_accuracy(GradientBoostingClassifier, X_train, y_train)


# In[ ]:

# [[tn, fp]]
# [[fn, tp]]
confusion_matrix(y, y_pred)

