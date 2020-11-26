import pandas as pd 
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dataset = pd.read_csv('Datasets/RPDR_Predictor_Dataset.csv')
data = dataset[['age_n','Wins_n','High_n','Safe_n', 'Low_n', 'Bottom_n', 'Lat', 'Lng']]
label  = dataset[['Label']]

X = data.to_numpy()
y = np.ravel(label.to_numpy())



clf = LinearDiscriminantAnalysis()
X_transform  = clf.fit_transform(X,y)

#plt.scatter(X_transform[:,0],X_transform[:,1])
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.25, random_state=42)

'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = LinearDiscriminantAnalysis()
clf.fit(X_train,y_train)

X_trainT = clf.transform(X_train)
X_testT = clf.transform(X_test)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_predict = neigh.predict(X_test)
'''

detree = DecisionTreeClassifier()
detree.fit(X_train, y_train)
y_predict = detree.predict(X_test)

print(confusion_matrix(y_test, y_predict))
print(accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

#print(x)
#print(y_test)