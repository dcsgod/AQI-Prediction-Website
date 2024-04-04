import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg1 = LinearRegression()
reg1.fit(x_train, y_train)
pred1 = reg1.predict(x_test)

reg2 = Lasso()
reg2.fit(x_train, y_train)
pred2 = reg2.predict(x_test)

reg3 = Ridge()
reg3.fit(x_train, y_train)
pred3 = reg3.predict(x_test)

reg4 = DecisionTreeRegressor()
reg4.fit(x_train, y_train)
pred4 = reg4.predict(x_test)
'''
print("Model\t\t\t RootMeanSquareError \t\t Accuracy of the model")
print("""Linear Regression \t\t {:.4f} \t \t\t {:.4f}""".format(np.sqrt(mean_squared_error(y_test, pred1)), reg1.score(x_train, y_train)))
print("""Lasso Regression \t\t {:.4f} \t \t\t {:.4f}""".format(np.sqrt(mean_squared_error(y_test, pred2)), reg2.score(x_train, y_train)))
print("""Ridge Regression \t\t {:.4f} \t \t\t {:.4f}""".format(np.sqrt(mean_squared_error(y_test, pred3)), reg3.score(x_train, y_train)))
print("""Decision Tree Regressor\t\t {:.4f} \t \t\t {:.4f}""".format(np.sqrt(mean_squared_error(y_test, pred4)), reg4.score(x_train, y_train)))
'''
# Define a function to make predictions using the trained model
def predict_aqi(pm25, pm10, no, no2, co, so2, o3, model):
    # Reshape input into a 2D array to match the model's expectations
    input_data = np.array([[pm25, pm10, no, no2, co, so2, o3]])
    # Make prediction using the specified model
    prediction = model.predict(input_data)
    return prediction[0]


# Assuming reg4 is the trained Decision Tree Regressor model
pm25_input = float(input("Enter PM2.5 value: "))
pm10_input = float(input("Enter PM10 value: "))
no_input = float(input("Enter NO value: "))
no2_input = float(input("Enter NO2 value: "))
co_input = float(input("Enter CO value: "))
so2_input = float(input("Enter SO2 value: "))
o3_input = float(input("Enter O3 value: "))

prediction = predict_aqi(pm25_input, pm10_input, no_input, no2_input, co_input, so2_input, o3_input, reg4)
print("Predicted AQI:", prediction)
