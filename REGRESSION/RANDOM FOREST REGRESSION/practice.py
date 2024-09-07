import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('REGRESSION/RANDOM FOREST REGRESSION/Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

model = RandomForestRegressor(n_estimators=30, random_state=0)
model.fit(x, y)

print(model.predict([[6.5]]))