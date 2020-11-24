import pandas as pd 

rankings = pd.read_csv('Datasets/RuPaul-Predict-A-Looza — Data Tables - all_rankings.csv')
contestants = pd.read_csv('Datasets/RuPaul-Predict-A-Looza — Data Tables - all_contestants.csv')

dataset = contestants[['contestant_id', 'season_number', 'age', 'hometown_city', 'hometown_state', 'season_outcome']]

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


dataset.insert(5, 'Wins', wins)
dataset.insert(6, 'Highs', highs)
dataset.insert(7, 'Safe', safe)
dataset.insert(8, 'Low', low)
dataset.insert(9, 'Bottom', btm)
dataset.insert(10, 'Number_of_Episodes', n)

## labels 
contestants_per_szn_dct = {}
for s in dataset.season_number.unique():
	contestants_per_szn_dct[s] = len(dataset[dataset['season_number']==s])


#smaller value means they performerd better
fraction_label = dataset.season_outcome/[contestants_per_szn_dct[x] for x in dataset.season_number]
print(fraction_label)

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


dataset.insert(12, 'Label', labels)
dataset.to_csv('Datasets/RPDR_Predictor_Dataset.csv', index=False)