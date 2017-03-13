import json
import sys
import csv
from math import *
import pprint
import time
import datetime
import numpy as np

import requests
from requests import get
from bs4 import BeautifulSoup
import bs4

from collections import defaultdict

import geocoder

# =======================================================
# These functions manually scrape through airbnb webpages
# =======================================================
def get_current_listings_by_zipcode(zipcode_list, city, state):
    '''
    INPUT:  list of zipcodes
    OUTPUT: list of airbnb listing urls
    '''

    listing_search_list=[]
    lat_lng_bound_list=[]

    for zipcode in zipcode_list:
        g = geocoder.google(zipcode)

        # Get lat-long bounds for zipcode
        # This uses Google's geocoder module
        laglng_bound = (g.bbox['southwest'][0], g.bbox['northeast'][0],
                        g.bbox['southwest'][1], g.bbox['northeast'][1])

        print "Searching Airbnb for zipcode {}, latlng_bound = {}".format(zipcode,laglng_bound)
        listing_search_list.append(get_urls_from_airbnb(city, state, zipcode, laglng_bound))
        lat_lng_bound_list.append(laglng_bound)
        time.sleep(3+np.random.random()*3)

    return listing_search_list

def get_urls_from_airbnb(city, state, zipcode, latlng_bound=(47.6185289, -122.3219921,47.60317389999999, -122.351255), max_pages=True):
    # https://www.airbnb.com/s/Seattle--WA?page=2
    '''
    Description:
    Queries airbnb for listings based on city, state, and lat-long bounds
    and returns a list of urls to each listing

    INPUT:  str, str, int, tuple, boolean
    OUTPUT: list
    '''

    # Initializing
    city0 = city.lower()[0]
    link_list = []
    sw_lat, ne_lat, sw_lng, ne_lng = latlng_bound
    page_no=0
    url = 'https://www.airbnb.com/{}/{}--{}?page={}&room_types%5B%5D=Private%20room&allow_override%5B%5D=&ne_lat={}&ne_lng={}&sw_lat={}&sw_lng={}&search_by_map=true'.format(city0,city,state,page_no,ne_lat,ne_lng,sw_lat,sw_lng)
    print url

    # Get first page
    response = get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    response.close()

    # Count number of results pages
    next_link_cnt = len(soup.findAll('a', attrs={'rel':'next'}))

    # Get all url links on first page
    for link in soup.findAll('a', attrs={'rel':'noopener'}):

        if 'room' in str(link.get('href')):
            #print link
            link_list.append(link.get('href'))

    # If more than one page of results, cycle through each page
    if (max_pages) & (next_link_cnt > 1):
        for page_no in range(1,next_link_cnt):
            time.sleep(5+np.random.random()*10) # delays for few seconds
            url = 'https://www.airbnb.com/{}/{}--{}?page={}&room_types%5B%5D=Private%20room&allow_override%5B%5D=&ne_lat={}&ne_lng={}&sw_lat={}&sw_lng={}&search_by_map=true'.format(city0,city,state,page_no,ne_lat,ne_lng,sw_lat,sw_lng)
            print url
            response = get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            response.close()

            # Get all url links on results page
            for link in soup.findAll('a', attrs={'rel':'noopener'}):
                if 'room' in str(link.get('href')):
                    #print link
                    link_list.append(link.get('href'))

    # Append filename with date
    datestring = '{:04d}{:02d}{:02d}'.format(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
    filename_today = 'data/{}_{}_query_{}.txt'.format(city,zipcode, datestring)

    # Write urls to file
    with open(filename_today, mode='a') as outfile:
        print "Saving results to file: ", filename_today
        for link in link_list:
            outfile.write("https://www.airbnb.com{}\n".format(link))

    return link_list

def get_listing_info(soup):
    '''
    Description: Parse through soup object and scrape information

    INPUT: BeautifulSoup object
    OUTPUT: dictionary
    '''
    info_dict = defaultdict()

    for i,div in enumerate(soup.findAll(['div','strong'], attrs={'class': 'col-md-6'})):
        for content in div.contents:
            if (type(content) == bs4.element.Tag):
                line = content.text
                if (not line.isspace()) & (':' in line):
                    #print "{}".format(content.text.strip())
                    key = line.split(':')[0].encode('ascii', errors='ignore')
                    val = line.split(':')[1].encode('ascii', errors='ignore')
                    info_dict[key] = val

    return info_dict

def scrape_data_from_urls(url_list,time_delay=15):
    '''
    Description: Iterate through airbnb url_list and scrape relevant information from page

    INPUT: list (of urls), int (time delay)
    OUTPUT: None
    '''

    # Append date string to filename
    datestring = '{:04d}{:02d}{:02d}'.format(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
    out_filename = 'data/scraped_listing_info_{}.txt'.format(datestring)

    print "Writing data to: {}".format(out_filename)

    # Open file for writing data
    with open(out_filename, mode='a') as outfile:

        # Initialize dictionary for page info
        page_info=defaultdict(list)

        # Iterate through url list
        for i,url in enumerate(url_list):
            time.sleep(time_delay+np.random.random()*15) # delays for few seconds
            response = get(url)

            if response.status_code != 200:
                descriptions.append((i,"Error"))
            else:
                # Create BeautifulSoup object with html.parser
                soup = BeautifulSoup(response.content, 'html.parser')

                # Get property id
                prop_id = url.split('/')[-1]

                # Get listing title
                title = soup.title.string.encode('utf-8', errors='ignore')

                # Write data to file
                outfile.write("prop_id: {}\n".format(prop_id))

                # If listing is active
                if (title.split()[0] != 'Airbnb:') & (title.split()[0] !='Vacation'):

                    # Get listing description
                    description = soup.findAll('p')[1].text.encode('utf-8', errors='ignore')

                    # Get listing information
                    info = get_listing_info(soup)

                    # Get metadata
                    metadata = soup.findAll('meta', attrs={'id':'_bootstrap-room_options'})
                    d_meta = json.loads(str(metadata[0].get('content').encode('utf-8', errors='ignore')))
                    d_meta = d_meta['airEventData']

                    # Write data to file
                    outfile.write("title: {}\n".format(title))
                    outfile.write("description: {}\n".format(description))

                    # Get number of reviews
                    num_reviews_bs = soup.findAll('h4', attrs={'class':'col-middle va-bottom review-header-text text-center-sm'})
                    if len(num_reviews_bs)>0:

                        num_reviews = num_reviews_bs[0].text
                        outfile.write("num_reviews: {}\n".format(num_reviews))

                        # Determine how many full stars and half stars
                        half_stars = soup.findAll('i', attrs={'class':'icon-star-half icon icon-babu icon-star-big'})
                        full_stars = soup.findAll('i', attrs={'class':'icon-star icon icon-babu icon-star-big'})

                        # If half rating
                        if len(half_stars)>0:
                            avg_rating = 0.5 + float(len(full_stars))
                        else:
                            avg_rating = float(len(full_stars))

                        # Write data to file
                        outfile.write("avg_rating: {}\n".format(avg_rating))

                    for key2, value2 in info.items():

                        # Write listing information data to file
                        outstring = "{}: {}\n".format(str(key2).strip(), str(value2).strip())
                        outfile.write(outstring)

                    for key3, value3 in d_meta.items():

                        # Write meta data to file
                        outstring = "{}: {}\n".format(str(key3).strip(), str(value3).strip())
                        outfile.write(outstring)

                # Listing is not active, fill with 'NA'
                else:
                    title='NA'
                    description = 'NA'
                    info = 'NA'
                    outfile.write("title: {}\n".format(title))
                    outfile.write("description: {}\n".format(description))
                    outfile.write("details: {}\n".format(info))

                print ("Iteration {}: prop_id {}, url {}".format(i, prop_id, url))
                page_info[prop_id] = [title, description, info]

            # Close response
            response.close()

# =======================================================
# These functions use the airbnb api
# =======================================================
def getrequest_listing_search_api(offset, price_min, price_max, lat, lng):
    url = 'https://api.airbnb.com/v2/search_results'

    #refer to http://airbnbapi.org/#listing-search


    payload = {
        'client_id':'3092nxybyb0otqw18e8nh5nty',
        'locale':'en-US',
        'currency':"USD",
        '_format':"for_search_results",
        '_limit':50,
        '_offset':offset,
        'guests':1,
        'ib': "false",
        'ib_add_photo_flow':"true",
        'min_bathrooms':0,
        'min_bedrooms':	1,
        'max_bedrooms':10,
        'min_beds':	1,
        'price_min': price_min,
        'price_max': price_max,
        'min_num_pic_urls':	1,
        'sort':	1,
        'suppress_facets': "true",
        'user_lat':	lat,
        'user_lng':	lng,
        'location': "Seattle, WA"

    }

    r = requests.get(url, params=payload)
    print(r.url)
    if r != None:
        flag = True
        #print(r.json())
        return r.json(), flag

    else:
        flag = False
    #print(r.text)
    return None, flag

def getrequest_listing_info_api(listing_id):
    url = 'https://api.airbnb.com/v2/listings/{}'.format(listing_id)

    #refer to http://airbnbapi.org/#listing-search

    payload = {
        'client_id':'3092nxybyb0otqw18e8nh5nty',
        '_format':'v1_legacy_for_p3',
        'locale':'en-US'
    }

    r = requests.get(url, params=payload)
    print(r.url)
    if r != None:
        flag = True
        #print(r.json())
        return r.json(), flag

    else:
        flag = False
    #print(r.text)
    return None, flag
