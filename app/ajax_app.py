from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
import ajax_app_helper as hlp
import datetime

app = Flask(__name__)


with open('static/model_key_model_merge_gdbr1.pkl') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET'])
def index():
    """Render a splash page containing input fields where the user can input
    lat-long coordinates"""
    return render_template('main/index.html')

@app.route("/build-chart", methods=['POST', 'GET'])
def build_chart():
    id = request.form['id']
    table = clean_dataframe(vertica_query(id))
    chart_json = parse_to_json(table)

    return render_template('render.html', data=table, chart_json=chart_json)

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the lat-long inputs use the model to predict airbnb income
    """

    # Get user provided lat-lng
    user_data = request.json
    latitude,longitude = user_data['lat'], user_data['lng']


    # Setting values for inputs, default to mean values
    days_since_created = 0.0  # Setting to 0.0
    number_of_photos = 16.1393133202  # Assuming average number of photos
    instantbook_enabled = 0.0 # Assuming no instabook to start

    superhost = 0.0  # Assuming no superhost to start
    overall_rating = 0.0 # Assuming no reviews to start
    number_of_reviews = 0.0 # Assuming no reviews to start
    pike_market = 0.0  # Assuming not in pike-place
    nonroom = 0.0  # Assuming an actual room
    private_bath = 0.100125831386  # Initializing to mean value, will be user defined
    view = 0.0648930433219   # Initializing to mean value, will be user defined
    water = 0.00970699262988  # Initializing to mean value, will be user defined
    parking = 0.00988675175265   # Initializing to mean value, will be user defined

    # initialing all property types to 0.0, will be user defined
    apt = 1.0
    bnb = 0.0
    cnd = 0.0
    hse = 0.0
    lft = 0.0
    oth = 0.0
    twn = 0.0

    view_user = user_data['view']
    bath_user = user_data['bath']

    print "view_user = ", view_user
    print "bath_user = ", bath_user
    #prop_type_user = user_data['prop_type']

    if view_user=='yes':
        print "====== We have a view!!!"
        view=1.0
    else:
        view=0.0
    if bath_user=='yes':
        print "====== We have a private bath!!!"
        private_bath=1.0
    else:
        private_bath=0.0

    # input_data_base = np.array((latitude, longitude ,0,0,0,0,0,0,0,0,0,0,0,0,apt,bnb,cnd,hse,lft,oth,twn)).reshape(1,-1)

    input_data_base = np.array((days_since_created, number_of_photos, instantbook_enabled,
                                latitude, longitude,
                                superhost, overall_rating, number_of_reviews,
                                pike_market, nonroom, private_bath, view, water, parking,
                                0,0,0,0,0,0,0,0,0,0,0,0,
                                apt,bnb,cnd,hse,lft,oth,twn)).reshape(1,-1)

    # Adding month flag for each iteration
    print "Assigning months..."
    #print input_data_base
    input_data_list=[]
    print input_data_base.shape
    for i in range(12):
        month_input_data = np.copy(input_data_base)
        month_input_data[:,i+14]=1
        input_data_list.append(month_input_data)

    pred_list_int=[]

    print "Getting predictions..."
    for input_data in input_data_list:
        pred = int(model.predict(input_data)[0])
        pred_list_int.append(int(pred))

    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Save figure
    # img_filepath = hlp.save_bar_chart_figure(pred_list_int,x_labels)

    # Save results to csv
    pred_df = pd.DataFrame(zip(month_list, pred_list_int), columns=['month', 'prediction']).set_index('month')
    # pred_df.to_csv('static/pred.csv')
    # pred_df.to_json('static/pred_new.json')

    json_list = []
    for month, pred in zip(month_list, pred_list_int):
        d = {}
        d['month'] = month
        d['prediction'] = pred
        json_list.append(d)

    print pred_list_int
    return jsonify(json_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
