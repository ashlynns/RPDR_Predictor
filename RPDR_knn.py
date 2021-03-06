import pandas as pd 
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#dataset = pd.read_csv('Datasets/RPDR_Predictor_Dataset_test.csv')
#data = dataset[['age_n','High_n','Safe_n', 'Low_n', 'Lat', 'Lng']]
dataset = pd.read_csv('Datasets/RPDR_Predictor_Dataset.csv')
data = dataset[['age_n','Wins_n','High_n','Safe_n', 'Low_n', 'Bottom_n', 'Lat', 'Lng']]
#data = dataset[['age_n','Wins_n','High_n','Safe_n', 'Low_n', 'Bottom_n', 'Lat', 'Lng']]
label  = dataset[['Label']]

X = data.to_numpy()
y = np.ravel(label.to_numpy())

clf = LinearDiscriminantAnalysis(n_components=2)
X_transform  = clf.fit_transform(X,y)

X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.20, random_state=42)

print(X_train)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_predict = neigh.predict(X_test)

print(confusion_matrix(y_test, y_predict))
print(accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))