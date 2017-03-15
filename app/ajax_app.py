from flask import Flask, render_template, request, jsonify
import pickle
from build_model import TextClassifier

app = Flask(__name__)


with open('static/model_gdbr0.pkl') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET'])
def index():
    """Render a splash page containing input fields where the user can input
    lat-long coordinates"""
    return render_template('main/index.html')

@app.route('/predict', methods=['GET'])
def predict():
    """Recieve the lat-long inputs use the model to predict airbnb income
    """
    user_data = request.json
    lag,lng = user_data['lat'], user_data['lng']
    pred = model.predict(np.array(lat,lng))
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
