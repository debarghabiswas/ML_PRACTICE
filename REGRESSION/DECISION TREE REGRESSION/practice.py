import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('REGRESSION/DECISION TREE REGRESSION/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

model = DecisionTreeRegressor(random_state=0)
model.fit(x, y)

model_pred = model.predict([[6.5]])
print(f"Predicted value of {6.5} is {model_pred[0]}")