#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#IMPORTING THE DATASET
dataset = pd.read_csv('D:/DEBARGHA BISWAS/MACHINE LEARNING COURSE (UDEMY)/REGRESSION/SUPPORT VECTOR REGRESSION/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#RESHAPING THE Y DEPENDENT MATRICS SINCE WE HAVE TO APPLY FEATURE SCALING 
y = y.reshape(len(y), 1)

#USING STANDARDISATION SCALING METHOD
sc_x = StandardScaler()#STANDARDISATION SCALING METHOD FOR THE X FEATURE MATRICS
sc_y = StandardScaler()#STANDARDISATION SCALING METHOD FOR THE Y MATRICS
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#USING THE KERNEL RADICAL BASIS FUNCTION (RBF) FOR SVR REGRESSION MODEL
model = SVR(kernel = 'rbf')
model.fit(x, y)

#CREATING THE PREDICTING DATA OF 2D DIMENSION
pred_data = np.array([[6.5]])
'''HERE WE USE THE INVERVE TRANSFORM METHOD SINCE THE OUTPUT OF THE PREDICT VALUE IS IN STANDARDISATION FORM. THIS WILL INVERSE INTO THE REAL VALUE.
    RESHAPING IS USE HERE SINCE WE HAVE TO USE THE INVERSE TRANSFORM METHOD WHICH ONLY ACCEPT 2D ARRAYS
'''
model_pred = sc_y.inverse_transform(model.predict(sc_x.transform(pred_data)).reshape(-1,1))

#PRINTING THE PREDICTED VALUE
print(f"Predicted value for {pred_data[0][0]} is {model_pred[0][0]}")