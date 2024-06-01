from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import pickle
from keras.models import load_model

app = Flask(__name__)

# Load your scaler and model
scaler = joblib.load('scaler.pkl')
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        room_number = int(request.form['roomNumber'])
        total_beds = int(request.form['totalBeds'])
        busy_beds = int(request.form['busyBeds'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        # Create the DataFrame
        prediction_data = pd.DataFrame({
            'Room Number': [room_number],
            'Total Beds': [total_beds],
            'Busy Bees': [busy_beds],
            'month': [month],
            'day': [day],
        })

        # Scale the input data
        prediction_scaled = scaler.transform(prediction_data)

        # Make prediction
        prediction = model.predict(prediction_scaled)
        predicted_class = int(np.argmax(prediction))  # Convert to int here

        # Return prediction as JSON response
        return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port =80)
