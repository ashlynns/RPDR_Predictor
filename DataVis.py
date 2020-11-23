import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Datasets/RPDR_Predictor_Dataset.csv')
L1 = dataset[dataset['Label']==1]
L2 = dataset[dataset['Label']==2]
L3 = dataset[dataset['Label']==3]
L4 = dataset[dataset['Label']==4]

