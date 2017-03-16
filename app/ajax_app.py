from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
app = Flask(__name__)


with open('static/cl_gdbr_monthly_model.pkl') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET'])
def index():
    """Render a splash page containing input fields where the user can input
    lat-long coordinates"""
    return render_template('main/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the lat-long inputs use the model to predict airbnb income
    """
    user_data = request.json
    lat,lng = user_data['lat'], user_data['lng']
    #months = [1,2,3,4,5,6,7,8,9,10,11,12]
    apt = 1
    bnb = 0
    cnd = 0
    hse = 0
    lft = 0
    oth = 0
    twn = 0

    input_data_base = np.array((lat, lng,0,0,0,0,0,0,0,0,0,0,0,0,apt,bnb,cnd,hse,lft,oth,twn)).reshape(1,-1)
    input_data_list=[]
    for i in range(12):
        month_input_data = np.copy(input_data_base)
        month_input_data[:,i+2]=1
        input_data_list.append(month_input_data)

    pred_list=[]
    pred_list_int=[]

    for input_data in input_data_list:
        pred = int(model.predict(input_data)[0])
        pred_list.append(str(pred))
        pred_list_int.append(int(pred))

    pred_series = pd.Series.from_array(pred_list_int)

    # Saving plot to file
    cgfont = {'fontname':'Century Gothic'}
    x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # now to plot the figure...
    plt.figure(figsize=(7, 4))
    ax = pred_series.plot(kind='bar', rot=0, grid=None, fontsize=13, color="#0099DF", width=0.7)
    ax.set_xticklabels(x_labels,**cgfont)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    rects = ax.patches
    labels = pred_list

    for rect, label in zip(rects, pred_list):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 1, label, ha='center', va='bottom', fontsize=16, **cgfont)

    plt.yticks([])

    savefig('templates/main/monthly_income', dpi=None, facecolor='w', edgecolor=None,
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.0,
        frameon=None)

    out_pred=[]
    for mo, pred in zip(x_labels, pred_list):
        out_str = "{}: {}".format(mo, pred)
        out_pred.append(out_str)

    out_pred = '\n'.join(out_pred)
    return jsonify({'prediction': out_pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
