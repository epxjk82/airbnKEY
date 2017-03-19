import json
import sys
import time, datetime
import numpy as np
import pandas as pd
from collections import defaultdict

def import_txtfile_as_df(filepath):
    """Convert data in txt files to pandas dataframe object

    INPUT:
        filepath: str (path to txt file with data)
        Note: Text files are from scrape_data_from_urls function from airbnb_scraping_helper.py
    RETURNS:
        pandas df
    """

    datestamp0 = filepath.split('/')[-1].strip()
    datestamp = ''.join([(char) for char in datestamp0 if char.isdigit()])

    d=defaultdict()
    d2=defaultdict()

    #Ex.  filepath = 'data/airbnb_scraping/scraped_listing_info_20170313_ALL.txt'
    with open(filepath) as f:
        for i,line in enumerate(f):

            # For first line, initialize prop_Id
            if i==0:
                cur_prop_id = line.split(':')[1].strip()

            else:
                if ':' in line:
                    key = line.split(':')[0].strip()
                    val = line.split(':')[1].strip()
                    if key=='prop_id':
                        d[cur_prop_id] = d2
                        cur_prop_id = val
                        d2=defaultdict()
                    else:
                        #print key, val
                        d2[key] = val
                # If line does not contain ":", then there is an unwanted line break
                # Append this new line to previous key's value
                else:
                    print "Line break error on line ", i
                    d2[key] = d2[key] + ' ' + line

    df = pd.DataFrame.from_dict(d, orient='index', dtype=None)
    df=df.reset_index()
    df['url'] = df['index'].apply(lambda x: 'https://www.airbnb.com/rooms/{}'.format(x))
    df['datestamp'] = datestamp
    df.rename(columns={"index": "prop_id"}, inplace=True)
    return df

def convert_json_amenities_to_df(json_data):

    id_d={}
    for i,item in enumerate(json_data):
        listing_id = json_data[i]['id']

        # Amenities dictionary
        new_d = {}
        for d in json_data[i]['listing_amenities']:
            new_d[d['name']]=d['is_present']

        id_d[listing_id] = new_d

    df = pd.DataFrame.from_dict(id_d, orient='index')
    return df

def convert_json_price_to_df(json_data):

    id_d={}
    for i, item in enumerate(json_data):
        listing_id = json_data[i]['id']
        new_d={}
        for key, val in json_data[i]['price_interface'].iteritems():
            #print key, type(val)
            if type(val) is dict:
                new_d[key]=val['value']
            else:
                new_d[key]=val

            id_d[listing_id] = new_d

    df = pd.DataFrame.from_dict(id_d, orient='index')
    return df

def convert_json_desc_to_df(json_data):

    id_d={}
    for i, item in enumerate(json_data):
        listing_id = json_data[i]['id']
        new_d={}
        if json_data[i]['sectioned_description']:
            for key, val in json_data[i]['sectioned_description'].iteritems():
                #print key, type(val)
                if type(val) is dict:
                    new_d[key]=val['value']
                else:
                    new_d[key]=val

                id_d[listing_id] = new_d

    df = pd.DataFrame.from_dict(id_d, orient='index')
    return df
