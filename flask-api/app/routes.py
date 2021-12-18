from app import app
from app.controller import PredictController

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    return PredictController.predict()