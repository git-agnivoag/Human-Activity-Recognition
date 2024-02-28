import json
from flask import Flask, render_template, jsonify, request
import joblib

# Define the directory where the model is stored
directory = "C:/Users/abgho/Desktop/data"
model = joblib.load(directory + '/total_model.pkl')

app = Flask(__name__)
prediction = 'static'  # Default prediction

@app.route('/model_predict')
def model_predict():
    return jsonify(result=prediction)

@app.route('/', methods=['POST'])
def get_feature():
    global prediction
    if request.json:
        feat = json.loads(request.json)
        prediction = model.predict(feat.reshape(1, -1))  # Assuming model is a scikit-learn model
        print(prediction)
    return 'received'

@app.route('/')
def index():
    return render_template('model_predict.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
