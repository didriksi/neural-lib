import numpy as np
import matplotlib.pyplot as plt
import neural
import pandas as pd

ALF = "ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ\'".lower()

# Encode words into relative frequency of 29 letters in array of length 30
def encode(word):
	arr = [0]*len(ALF)
	for letter in word:
		arr[ALF.find(letter)] += 1/len(word)
	return arr

if __name__ == '__main__':

	# Make list of dataframes with encoded words, one for each language
	dataframes = []
	langlist = ['english', 'norwegian']
	for language in langlist:
		# This only works if you have csv-files with words in the languages you want, in datasets/
		wordData = pd.read_csv('datasets/' + language + '.csv')
		wordData[language] = 1
		dataframes.append(wordData)

	# Remove entries from the biggest set so that they both have the same length
	for i in range(1, len(dataframes)):
		if len(dataframes[i]) > len(dataframes[i-1]):
			dataframes[i] = dataframes[i].sample(n = len(dataframes[i-1]))
		else:
			dataframes[i-1] = dataframes[i-1].sample(n = len(dataframes[i]))

	# Concatenate all the dataframes, and aggregate the language columns so that words that are allowed in several languages are combined
	complete = pd.concat([dataframe for dataframe in dataframes], sort=False, join='outer', ignore_index=True)
	complete.set_index('word')
	complete = complete.groupby(['word']).agg('sum')
	complete['words'] = complete.index

	# Make the codes separate columns in the dataframe
	codes = np.zeros((complete['words'].size, len(ALF)))
	for e, word in enumerate(complete['words']):
		codes[e,:] = encode(word)
	for k, letter in enumerate(ALF):
		complete[letter] = codes[:,k]

	# Store in a csv file
	complete.to_csv('datasets/complete.csv', index=False)
