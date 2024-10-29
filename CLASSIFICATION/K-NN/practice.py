import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('CLASSIFICATION/K-NN/Social_Network_Ads.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)
print(np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1))

print("Confusion Matrics: \n", confusion_matrix(ypred, ytest))
print("Accuracy Score: ", accuracy_score(ypred, ytest))