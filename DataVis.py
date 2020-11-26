import matplotlib.pyplot as plt
import pandas as pd 
import scipy
from scipy import stats
import numpy 

dataset = pd.read_csv('Datasets/RPDR_Predictor_Dataset.csv')
total = len(dataset)
L1 = dataset[dataset['Label']==1]
L2 = dataset[dataset['Label']==2]
L3 = dataset[dataset['Label']==3]
L4 = dataset[dataset['Label']==4]

print("Class 1 size:", len(L1), (len(L1)/total)*100)
print("Class 2 size:", len(L2), (len(L2)/total)*100)
print("Class 3 size:", len(L3), (len(L3)/total)*100)
print("Class 4 size:", len(L4), (len(L4)/total)*100)

plt.hist(dataset.Label)
plt.show()
'''

b1 = [-0.5,0.5,1.5,2.5,3.5,4.5]
plt.hist(L5.Wins,rwidth=0.75, color = 'k', label = 'Class 5')
plt.hist(L4.Wins,rwidth=0.75, color = 'm', label = 'Class 4' )
plt.hist(L3.Wins,rwidth=0.75, color = 'r', label = 'Class 3')
plt.hist(L2.Wins,rwidth=0.75, color = 'g', label = 'Class 2' )
plt.hist(L1.Wins,rwidth=0.75, color = 'b' , label='Class 1')
plt.legend()
plt.show()

b2 = [0,0.1, 0.2, 0.3, 0.4]
plt.hist(L1.Wins/L1.Number_of_Episodes,bins=b2,rwidth=0.75, color = 'b' )
plt.hist(L2.Wins/L2.Number_of_Episodes,bins=b2,rwidth=0.75, color = 'g' )
plt.hist(L3.Wins/L3.Number_of_Episodes,bins=b2,rwidth=0.75, color = 'r' )
plt.hist(L4.Wins/L4.Number_of_Episodes,bins=b2,rwidth=0.75, color = 'm' )
plt.hist(L5.Wins/L5.Number_of_Episodes,bins=b2,rwidth=0.75, color = 'k' )
plt.show()

'''

def distributions(column, dataset):
	min_val = min(dataset[column])
	max_val = max(dataset[column])
	bins = numpy.linspace(min_val, max_val, 50)

	L1 = dataset[dataset['Label']==1]
	mu1, sigma1 = scipy.stats.norm.fit(L1[column])
	bfl1 = scipy.stats.norm.pdf(bins, mu1, sigma1)
	#plt.plot(bins, bfl1, label = 'Class 1', color='b')

	L2 = dataset[dataset['Label']==2]
	mu2, sigma2 = scipy.stats.norm.fit(L2[column])
	bfl2 = scipy.stats.norm.pdf(bins, mu2, sigma2)
	#plt.plot(bins, bfl2, label = 'Class 2', color = 'g')

	L3 = dataset[dataset['Label']==3]
	mu3, sigma3 = scipy.stats.norm.fit(L3[column])
	bfl3 = scipy.stats.norm.pdf(bins, mu3, sigma3)
	#plt.plot(bins, bfl3, label = 'Class 3', color = 'r')

	L4= dataset[dataset['Label']==4]
	mu4, sigma4 = scipy.stats.norm.fit(L4[column])
	bfl4 = scipy.stats.norm.pdf(bins, mu4, sigma4)
	#plt.plot(bins, bfl4, label = 'Class 4', color = 'm')

	L5 = dataset[dataset['Label']==5]
	mu5, sigma5 = scipy.stats.norm.fit(L5[column])
	bfl5 = scipy.stats.norm.pdf(bins, mu5, sigma5)
	#plt.plot(bins, bfl5, label = 'Class 5', color = 'k')

	#plt.legend()
	#plt.title(column)
	#plt.show()
	#return()
	return(bins, bfl1, bfl2, bfl3, bfl4)

def distributions_norm(column, dataset):
	bins = numpy.linspace(-1, 1, 50)

	L1 = dataset[dataset['Label']==1]
	mu1, sigma1 = scipy.stats.norm.fit(L1[column])
	bfl1 = scipy.stats.norm.pdf(bins, mu1, sigma1)
	#plt.plot(bins, bfl1, label = 'Class 1', color='b')

	L2 = dataset[dataset['Label']==2]
	mu2, sigma2 = scipy.stats.norm.fit(L2[column])
	bfl2 = scipy.stats.norm.pdf(bins, mu2, sigma2)
	#plt.plot(bins, bfl2, label = 'Class 2', color = 'g')

	L3 = dataset[dataset['Label']==3]
	mu3, sigma3 = scipy.stats.norm.fit(L3[column])
	bfl3 = scipy.stats.norm.pdf(bins, mu3, sigma3)
	#plt.plot(bins, bfl3, label = 'Class 3', color = 'r')

	L4= dataset[dataset['Label']==4]
	mu4, sigma4 = scipy.stats.norm.fit(L4[column])
	bfl4 = scipy.stats.norm.pdf(bins, mu4, sigma4)
	#plt.plot(bins, bfl4, label = 'Class 4', color = 'm')

	L5 = dataset[dataset['Label']==5]
	mu5, sigma5 = scipy.stats.norm.fit(L5[column])
	bfl5 = scipy.stats.norm.pdf(bins, mu5, sigma5)
	#plt.plot(bins, bfl5, label = 'Class 5', color = 'k')

	#plt.legend()
	#plt.title(column+" norm")
	#plt.show()
	
	return(bins, bfl1, bfl2, bfl3, bfl4)

def subplots(column, column_n,  dataset): 
	bins, bfl1, bfl2, bfl3, bfl4 = distributions(column, dataset)
	nbins, nbfl1, nbfl2, nbfl3, nbfl4 = distributions_norm(column_n, dataset)

	plt.figure(1)

	plt.subplot(211)
	plt.plot(bins, bfl1, label = 'Class 1', color='k', ls = '-')
	plt.plot(bins, bfl2, label = 'Class 2', color = 'k', ls = '--')
	plt.plot(bins, bfl3, label = 'Class 3', color = 'k', ls = '-.')
	plt.plot(bins, bfl4, label = 'Class 4', color = 'k', ls = ':')
	#plt.plot(bins, bfl5, label = 'Class 5', color = 'k', marker = '.')
	plt.legend()
	plt.title("Distribution of Queens Age per Class \n Pre Normalization")
	#plt.title("Distribution of "+column+" Queens per Class \n Pre Normalization")

	plt.subplot(212)
	plt.plot(nbins, nbfl1, label = 'Class 1', color='k', ls = '-')
	plt.plot(nbins, nbfl2, label = 'Class 2', color = 'k', ls = '--')
	plt.plot(nbins, nbfl3, label = 'Class 3', color = 'k', ls = '-.')
	plt.plot(nbins, nbfl4, label = 'Class 4', color = 'k', ls = ':')
	#plt.plot(nbins, nbfl5, label = 'Class 5', color = 'k', marker='.')
	plt.legend()
	plt.title("Distribution of Queens Age per Class \n Post Normalization")
	#plt.title("Distribution of "+column+" Queens per Class \n Post Normalization")

	plt.show()

	return()
