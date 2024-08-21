import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#IMPORTING DATASET AND SEPARATING INTO INDEPENDENT VARIABLE AND DEPENDENT VARIABLE
dataset = pd.read_csv('D:\DEBARGHA BISWAS\MACHINE LEARNING COURSE (UDEMY)\DATA PREPROCESSING\Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#REPLACING THE MISSING VALUES WITH MEAN VALUE
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#ENCODING THE CATEGORICAL DATAS
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

le = LabelEncoder()
y = le.fit_transform(y)

#SPLITTING THE DATASET INTO TRAINING SET AND TEST SET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#APPLYING FEATURE SCALING
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])


print(x_train, "\n\n\n")
print(x_test)
