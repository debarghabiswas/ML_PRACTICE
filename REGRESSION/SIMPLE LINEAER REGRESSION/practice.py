import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('D:\DEBARGHA BISWAS\MACHINE LEARNING COURSE (UDEMY)\REGRESSION\SIMPLE LINEAER REGRESSION\Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=2)

regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

ypred = regressor.predict(xtest)


plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.scatter(xtrain, ytrain, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.xlabel('experience')
plt.ylabel('salary')


plt.subplot(1,2,2)
plt.scatter(xtest, ytest, color = 'red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.xlabel('experience')
plt.ylabel('salary')


plt.show()