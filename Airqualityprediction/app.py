from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

app = Flask(__name__)

# Load the trained model
def load_model():
    data = pd.read_csv('D:\MLprojects\Airqualityprediction\city_day.csv')

    data.fillna({
        'PM2.5': data['PM2.5'].mean(),
        'PM10': data['PM10'].mean(),
        'NO': data['NO'].mean(),
        'NO2': data['NO2'].mean(),
        'CO': data['CO'].mean(),
        'SO2': data['SO2'].mean(),
        'O3': data['O3'].mean(),
        'AQI': data['AQI'].mean()
    }, inplace=True)

    newdata = data.drop(['City', 'Date', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene', 'AQI_Bucket'], axis=1)

    x = newdata[['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'SO2', 'O3']]
    y = newdata['AQI']

    reg4 = DecisionTreeRegressor()
    reg4.fit(x, y)

    return reg4

model = load_model()

# Define a function to make predictions using the trained model
def predict_aqi(pm25, pm10, no, no2, co, so2, o3, model):
    # Reshape input into a 2D array to match the model's expectations
    input_data = np.array([[pm25, pm10, no, no2, co, so2, o3]])
    # Make prediction using the specified model
    prediction = model.predict(input_data)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pm25_input = float(request.form['pm25'])
    pm10_input = float(request.form['pm10'])
    no_input = float(request.form['no'])
    no2_input = float(request.form['no2'])
    co_input = float(request.form['co'])
    so2_input = float(request.form['so2'])
    o3_input = float(request.form['o3'])

    prediction = predict_aqi(pm25_input, pm10_input, no_input, no2_input, co_input, so2_input, o3_input, model)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
