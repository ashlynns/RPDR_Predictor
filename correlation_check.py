import pandas as pd 
import scipy
from scipy.stats import pearsonr


dataset = pd.read_csv('Datasets/RPDR_Predictor_Dataset.csv')
total = len(dataset)

L1 = dataset[dataset['Label']==1]
L2 = dataset[dataset['Label']==2]
L3 = dataset[dataset['Label']==3]
L4 = dataset[dataset['Label']==4]

def compute(c1,c2, dataset): 
	t1 = dataset[c1]
	t2 = dataset[c2]

	c12 = pearsonr(t1, t2)

	return (c12)


def correlate(dataset):
	t = compute('Wins', 'High', dataset)
	print('WinsxHigh=', t[0])
	t = compute('Wins', 'Safe', dataset)
	print('WinsxSafe=', t[0])
	t = compute('Wins', 'Low', dataset)
	print('WinsxLow=', t[0])
	t = compute('Wins', 'Bottom', dataset)
	print('WinsxBottom=', t[0])

	t = compute('High', 'Safe', dataset)
	print('HighxSafe=', t[0])
	t = compute('High', 'Low', dataset)
	print('HighxLow=', t[0])
	t = compute('High', 'Bottom', dataset)
	print('HighxBottom=', t[0])

	t = compute('Safe', 'Low', dataset)
	print('SafexLow=', t[0])
	t = compute('Safe', 'Bottom', dataset)
	print('SafexBottom=', t[0])
	
	t = compute('Low', 'Bottom', dataset)
	print('LowxBottom=', t[0])








### L1 ### 
print('CLASS 1 ')
correlate(dataset)
