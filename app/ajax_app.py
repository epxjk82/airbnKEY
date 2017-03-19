from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
import ajax_app_helper as hlp
import datetime

app = Flask(__name__)


with open('static/cl_gdbr_monthly_model.pkl') as f:
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
    lat,lng = user_data['lat'], user_data['lng']

    # Dummy codes for property type
    apt = 1
    bnb = 0
    cnd = 0
    hse = 0
    lft = 0
    oth = 0
    twn = 0

    input_data_base = np.array((lat, lng,0,0,0,0,0,0,0,0,0,0,0,0,apt,bnb,cnd,hse,lft,oth,twn)).reshape(1,-1)

    # Adding month flag for each iteration
    input_data_list=[]
    for i in range(12):
        month_input_data = np.copy(input_data_base)
        month_input_data[:,i+2]=1
        input_data_list.append(month_input_data)

    pred_list_int=[]

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
