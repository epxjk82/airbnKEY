from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


with open('static/model_gdbr0.pkl') as f:
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
    latlng_data = np.array((lat, lng)).reshape(1,-1)
    pred = int(model.predict(latlng_data)[0])
    #pred_dollar = "${}".format(pred)
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
