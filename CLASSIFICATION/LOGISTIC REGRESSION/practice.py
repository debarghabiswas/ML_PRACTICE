import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

#IMPORTING THE DATASET
data = pd.read_csv('CLASSIFICATION/LOGISTIC REGRESSION/Social_Network_Ads.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#SPLITTING THE DATASET INTO TRAINING AND TEST SET
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=10)

#APPLYING FEATURE SCALING
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

#FITTING THE DATASET INTO THE LOGISTIC REGRESSION MODEL
model = LogisticRegression(random_state=0)
model.fit(xtrain, ytrain)

#PREDICTING A NEW RESULTS
new_data = sc.transform([[30, 87000]])
data_pred = model.predict(new_data)
print("The Predicted value of the man of 30 years of old and estimated salary is 87,000",data_pred)

#FINDING THE CONFUSION MATRIX AND ACCURACY SCORE
ypred = model.predict(xtest)
cm = confusion_matrix(ytest, ypred)
ac = accuracy_score(ytest, ypred)
print("Confusion Matrics: \n", cm)
print("Accuracy Score: ", ac)