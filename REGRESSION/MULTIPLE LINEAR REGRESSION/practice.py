import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('D:/DEBARGHA BISWAS/MACHINE LEARNING COURSE (UDEMY)/REGRESSION/MULTIPLE LINEAR REGRESSION/50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state=5)

regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

np.set_printoptions(precision=2)
new_data = np.array([[1.0, 0.0, 0.0, 23640.93, 96189.63, 148001.11]])
pred_profit = regressor.predict(new_data)
print(f"Predicted Profit: {pred_profit}")