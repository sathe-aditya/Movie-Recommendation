from __future__ import print_function
import pandas as pd 
import numpy as np
import nltk, os, re, codecs, mpld3
from nltk.stem.snowball import SnowballStemmer
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib as mpl
df = pd.read_csv('movies.txt', sep='\t')

titles = df['Name']
plots = df['Plot']

totalPlots = []
totalTitles = []
print("Making totalTitles & totalPlots")
i = 0
j = 0
while(j<500):
	if(len(plots[i]) > 500):
		totalPlots.append(plots[i])
		totalTitles.append(titles[i])
		j = j + 1
	i = i + 1

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')

def tokenizeAndStem(text):
	tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filteredTokens = []
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filteredTokens.append(token)
	stems = [stemmer.stem(t) for t in filteredTokens]
	return stems

def tokenizeOnly(text):
	tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filteredTokens = []
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filteredTokens.append(token)
	return filteredTokens

totalVocabStemmed = []
totalVocabTokenized = []

print("Tokenizing")
for plot in totalPlots:
	allWordsStemmed = tokenizeAndStem(plot)
	totalVocabStemmed.extend(allWordsStemmed)
	allWordsTokenized = tokenizeOnly(plot)
	totalVocabTokenized.extend(allWordsTokenized)

vocabFrame = pd.DataFrame({'words':totalVocabTokenized}, index = totalVocabStemmed)

print("BOW")
vector = TfidfVectorizer(max_df = 0.8, min_df = 0.1, stop_words = 'english', use_idf = True, tokenizer = tokenizeAndStem, ngram_range = (1,3))
tfidfMatrix = vector.fit_transform(totalPlots)
terms = vector.get_feature_names()
print(terms)
dist = 1 - cosine_similarity(tfidfMatrix)

print("KMeans")

totalClusters = 6
clf = KMeans(n_clusters = totalClusters)
clf.fit(tfidfMatrix)
clusters = clf.labels_.tolist()
print (clf.cluster_centers_)

joblib.dump(clf, 'doc_cluster.pkl')
clf = joblib.load('doc_cluster.pkl')
clusters = clf.labels_.tolist()

books = {'title': totalTitles, 'plots':totalPlots, 'cluster':clusters}
frame = pd.DataFrame(books, index=[clusters], columns=['title','cluster'])

frame['cluster'].value_counts()

print('Top terms per cluster:')
print()

names = []
orderCentroids = clf.cluster_centers_.argsort()[:, ::-1]
for i in range(totalClusters):
	print('Cluster %d words:' % i, end='')

	for ind in orderCentroids[i, :6]:
		names.append(vocabFrame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8','ignore'))
		print(' %s' %vocabFrame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8','ignore'), end=',')
	print()
	print()

	print("Cluster %d titles:" % i, end='')
	for title in frame.ix[i]['title'].values.tolist():
		print(' %s,' %title,end='')
	print()
	print()

print("Plotting")
MDS()
mds = MDS(n_components=2, dissimilarity="precomputed",random_state=1)
pos=mds.fit_transform(dist)

xs, ys = pos[:, 0], pos[:, 1]

clusterColors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#45f12a', 6: '#a23c1b', 7: '#1cb2af'}
clusterNames = {0:names[0], 1:names[1], 2:names[2], 3:names[3], 4:names[4], 5:names[5], 6:names[6], 7:names[7]}

df = pd.DataFrame(dict(x=xs,y=ys,label=clusters,title=totalTitles))
groups = df.groupby('label')

css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

fig, ax = plt.subplots(figsize=(17,9))
ax.margins(0.05)

for name, group in groups:
	ax.plot(group.x, group.y, marker='o',linestyle='', ms=18, label=clusterNames[name],mec='none', color=clusterColors[name])
	ax.set_aspect('auto')

	ax.tick_params(\
		axis='x',
		which='both',
		bottom='off',
		top='off',
		labelbottom='off')
	ax.tick_params(\
		axis='y',
		which='both',
		left='off',
		top='off',
		labelleft='off')
	ax.legend(numpoints=1)

for i in range(len(df)):
	ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

centroids = clf.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], marker='x', color='b')
plt.show()