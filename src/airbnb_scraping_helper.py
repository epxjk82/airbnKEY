import requests
import json
import sys
import csv
from math import *
import pprint

def get_info(soup):
    info_dict=defaultdict()
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


def get_urls_from_airbnb(city, state):
# https://www.airbnb.com/s/Seattle--WA?page=2
    city0 = city.lower()[0]
    max_pages = 17

    link_list = []

    for page in xrange(max_pages+1):
        url = 'https://www.airbnb.com/{}/{}--{}?page={}&room_types%5B%5D=Private%20room'.format(city0,city,state,page)
        print url
        time.sleep(3+np.random.random()*3) # delays for few seconds
        response = get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        response.close()
        for link in soup.findAll('a', attrs={'class':'media-photo media-cover'}):
            link_list.append(link.get('href'))

    datestring = '{:04d}{:02d}{:02d}'.format(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
    filename_today = 'data/{}_query_{}.txt'.format(city, datestring)
    with open(filename_today, mode='a') as outfile:
        for link in link_list:
            print "Saving results to file: ", filename_today
            outfile.write("https://www.airbnb.com{}\n".format(link))
    return link_list

def scrape_data_from_urls(url_list):
    datestring = '{:04d}{:02d}{:02d}'.format(datetime.date.today().year, datetime.date.today().month, datetime.date.today().day)
    out_filename = 'data/scraped_listing_info_{}.txt'.format(datestring)

    print "Writing data to: {}".format(out_filename)
    with open(out_filename, mode='a') as outfile:
        page_info=defaultdict(list)
        for i,url in enumerate(url_list):
            time.sleep(5+np.random.random()*10) # delays for few seconds
            response = get(url)
            if response.status_code != 200:
                descriptions.append((i,"Error"))
            else:
                soup = BeautifulSoup(response.content, 'html.parser')
                prop_id = url.split('/')[-1]
                title = soup.title.string.encode('ascii', errors='ignore')
                outfile.write("prop_id: {}\n".format(prop_id))
                if (title.split()[0] != 'Airbnb:') & (title.split()[0] !='Vacation'):
                    description = soup.findAll('p')[1].text.encode('ascii', errors='ignore')
                    info = get_info(soup)
                    #print soup.findAll('span', attrs={'id':'book-it-price-string'})
                    #print soup.findAll('div', attrs={'class':'book-it__price-amount'})
                    #price = soup.findAll('div', attrs={'class':'book-it__price-amount'})[0].text.encode('ascii', errors='ignore')

                    outfile.write("title: {}\n".format(title))
                    outfile.write("description: {}\n".format(description))
                    #outfile.write("price: {}\n".format(price))

                    num_reviews_bs = soup.findAll('h4', attrs={'class':'col-middle va-bottom review-header-text text-center-sm'})
                    if len(num_reviews_bs)>0:
                        num_reviews = num_reviews_bs[0].text
                        outfile.write("num_reviews: {}\n".format(num_reviews))


                        half_stars = soup.findAll('i', attrs={'class':'icon-star-half icon icon-babu icon-star-big'})
                        full_stars = soup.findAll('i', attrs={'class':'icon-star icon icon-babu icon-star-big'})
                        if len(half_stars)>0:
                            print "half rating!"
                            avg_rating = 0.5 + float(len(full_stars))
                        else:
                            avg_rating = float(len(full_stars))
                        outfile.write("avg_rating: {}\n".format(avg_rating))
                    for key2, value2 in info.items():
                        #print ("{}: {}".format(str(key2)[2:-1].strip(), str(value2)[2:-1].strip()))
                        #outstring = "{}: {}\n".format(str(key2)[2:-1].strip(), str(value2)[2:-1].strip())
                        outstring = "{}: {}\n".format(str(key2).strip(), str(value2).strip())
                        outfile.write(outstring)
                else:
                    title='NA'
                    description = 'NA'
                    info = 'NA'
                    outfile.write("title: {}\n".format(title))
                    outfile.write("description: {}\n".format(description))
                    outfile.write("details: {}\n".format(info))
                print ("Iteration {}: prop_id {}, url {}".format(i, prop_id, url))
                page_info[prop_id] = [title, description, info]
            response.close()

def getrequest_listing_search(offset, price_min, price_max, lat, lng):
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
        'min_beds':	0,
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

def getrequest_listing_info(listing_id):
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
