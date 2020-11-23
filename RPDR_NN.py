import pandas as pd 
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('Datasets/RPDR_Predictor_Dataset.csv')
data = dataset[['nWins','nHighs','nSafe', 'nLow', 'nBottom']]
label  = dataset[['Label']]

X = data.to_numpy()
y = np.ravel(label.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

clf = MLPClassifier(max_iter=1000)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

print(confusion_matrix(y_test, y_predict))
print(accuracy_score(y_test, y_predict))
