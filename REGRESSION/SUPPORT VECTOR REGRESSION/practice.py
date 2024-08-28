import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('D:/DEBARGHA BISWAS/MACHINE LEARNING COURSE (UDEMY)/REGRESSION/SUPPORT VECTOR REGRESSION/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

model = SVR(kernel = 'rbf')
model.fit(x, y)

pred_data = np.array([[6.5]])
model_pred = sc_y.inverse_transform(model.predict(sc_x.transform(pred_data)).reshape(-1,1))

print(f"Predicted value for {pred_data[0][0]} is {model_pred[0][0]}")