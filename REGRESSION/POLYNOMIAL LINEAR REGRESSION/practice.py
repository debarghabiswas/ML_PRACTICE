import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('REGRESSION/POLYNOMIAL LINEAR REGRESSION/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

poly_reg = PolynomialFeatures(degree=7)
xpoly = poly_reg.fit_transform(x)#FITTING AND TRANSFORMING THE X FEATURE METRICS INTO POLYNOMIAL FEATURE
model = LinearRegression()
model.fit(xpoly, y)

new_data = np.array([[6.5]])#ALWAYS MAKE A 2D ARRAY FOR DATA
'''
SINCE THE MODEL IS A LINEAR REGRESSION THEREFORE WE HAVE TO FIRST TRANSFORM THE DATA 
INTO POLYNOMIAL FEATURE THAN WE CAN USE THAT NEW DATA INTO THE MODEL PREDICT METHOD
'''
new_data_poly = poly_reg.transform(new_data)
data_pred = model.predict(new_data_poly)

print(data_pred)