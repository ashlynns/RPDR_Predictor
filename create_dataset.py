import pandas as pd 

rankings = pd.read_csv('Datasets/RuPaul-Predict-A-Looza — Data Tables - all_rankings.csv')
contestants = pd.read_csv('Datasets/RuPaul-Predict-A-Looza — Data Tables - all_contestants.csv')
cities = pd.read_csv("Datasets/uscities.csv", usecols = ['city_ascii', 'state_name', 'lat', 'lng'])

dataset = contestants[['contestant_id', 'season_number', 'age', 'hometown_city', 'hometown_state', 'season_outcome']]

lat = []
lng = []

## retriving lat/long co-ordinates for hometowns
for i in range(0,len(dataset)):
	town = dataset.hometown_city.iloc[i]
	state = dataset.hometown_state.iloc[i]
	if town in cities.city_ascii.values: 
		subcities = cities[cities.city_ascii == town]
		if state in subcities.state_name.values: 
			t = cities[(cities['city_ascii'] == town) & (cities['state_name'] == state)]
			lat.append(t.lat.values[0])
			lng.append(t.lng.values[0])
dataset.insert(5, 'Lat', lat)
dataset.insert(6, 'Lng', lng) 

## 
def performance_stats(id):
	id_data = rankings[rankings['contestant_id']==id]
	n = len(id_data)
	wins = len(id_data[id_data['episode_placement']=='WIN'])
	high = len(id_data[id_data['episode_placement']=='HIGH'])
	safe = len(id_data[id_data['episode_placement']=='SAFE'])
	low = len(id_data[id_data['episode_placement']=='LOW'])
	bottom = len(id_data[id_data['episode_placement']=='BTM2']) + len(id_data[id_data['episode_placement']=='BTM6'])
	
	return(n, wins, high, safe, low, bottom)

n = []
wins = []
highs = []
safe = []
low = []
btm = []

for i in dataset.contestant_id: 
	n_id, win_id, high_id, safe_id, low_id, btm_id = performance_stats(i)
	n.append(n_id)
	wins.append(win_id)
	highs.append(high_id)
	safe.append(safe_id)
	low.append(low_id)
	btm.append(btm_id)

dataset.insert(7, 'Wins', wins)
dataset.insert(8, 'High', highs)
dataset.insert(9, 'Safe', safe)
dataset.insert(10, 'Low', low)
dataset.insert(11, 'Bottom', btm)
dataset.insert(12, 'Number_of_Episodes', n)

## labels 
contestants_per_szn_dct = {}
for s in dataset.season_number.unique():
	contestants_per_szn_dct[s] = len(dataset[dataset['season_number']==s])

#smaller value means they performed better
fraction_label = dataset.season_outcome/[contestants_per_szn_dct[x] for x in dataset.season_number]

labels = [] # This is written to divide into 4 performance categories 
for f in fraction_label: 
	if f <= 1/5:
		labels.append(1) # top quarter of queens - did best 
	if f > 1/5 and f<=2/5:
		labels.append(2) # 3rd quarter of queens - did well 
	if f > 2/5 and f <= 3/5: 
		labels.append(3) # 2nd quarter of queens - didnt do great 
	if f > 3/5 and f<=4/5: 
		labels.append(4) #1st quarter - did bad 
	if f > 4/5:
		labels.append(5)

dataset.insert(14, 'Label', labels)


### Normalization ### 

def minmaxnorm(data):
	minv = min(data)
	#print(minv)
	maxv = max(data)
	#print(maxv)
	scaled = (2*(data-minv)/(maxv-minv))-1 
	return(scaled)

def norm(column, dataset=dataset): 
	n = 'Number_of_Episodes'
	data = dataset[column]
	data_n1 = data/dataset[n] # normalized by episode apperances
	#print(data_n1.unique())
	data_n2 = minmaxnorm(data_n1) # min max scaling 
	return(data_n2)

#norm('Bottom')	

dataset.insert(3, 'age_n', norm('age'))
dataset.insert(9, 'Wins_n', norm('Wins'))
dataset.insert(11, 'High_n', norm('High'))
dataset.insert(13, 'Safe_n', norm('Safe'))
dataset.insert(15, 'Low_n', norm('Low'))
dataset.insert(17, 'Bottom_n', norm('Bottom'))

dataset.to_csv('Datasets/RPDR_Predictor_Dataset.csv', index=False)