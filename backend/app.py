import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from utils.fraud_prediction import FraudPrediction

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # print(data)
    fp = FraudPrediction()
    prediction = fp.predict(data)
    return jsonify({"prediction":f"{prediction}"})

if __name__ == '__main__':
    logging.basicConfig(filename='app.log', level=logging.INFO)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)