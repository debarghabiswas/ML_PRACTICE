import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('CLASSIFICATION/SVM/Social_Network_Ads.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()

xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

model = SVC(kernel='linear', random_state=0)
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

print(f"Confusion Matrics: \n{confusion_matrix(ytest, ypred)}")
print(f"Accuracy Score: {(accuracy_score(ytest, ypred))*100}%")