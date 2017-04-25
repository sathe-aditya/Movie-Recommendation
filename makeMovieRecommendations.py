import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import gensim
import itertools, os, pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities

'''
df1 = pd.read_csv('MovieSummaries/plot_summaries.txt', sep='\t', names=['ID','Plot'])
df2 = pd.read_csv('MovieSummaries/movie.metadata.tsv', sep='\t', names=['ID', 'ID1', 'Name', 'Date', 'Revenue', 'Runtime', 'Lang', 'Conutry', 'Genres'])

x = pd.merge(df1, df2, how='inner', on='ID')
x.to_csv('movies.txt', sep='\t')
'''

def extractNames(name, text):
	for sent in nltk.sent_tokenize(text):
		for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
			try:
				if chunk.label() == 'PERSON':
					for c in chunk.leaves():
						if str(c[0].lower()) not in name:
							name.append(str(c[0].lower()))
			except AttributeError:
				pass
	return name

df = pd.read_csv('movies.txt', sep='\t')
df = df[0:2000]

if (not os.path.isfile('index.p') and not os.path.isfile('lsi.p') and not os.path.isfile('corpusTfidf.p')):
	tokenizer = RegexpTokenizer(r'\w+')
	df['Plot'] = df['Plot'].apply(lambda x:' '.join(tokenizer.tokenize(x)))
	documents = list(df['Plot'])
	stopWords = set(stopwords.words('english'))

	names = []
	for doc in documents:
		names = extractNames(names, doc)
	stopWords.update(names)

	stemmer = SnowballStemmer('english')
	texts = [[stemmer.stem(word) for word in document.lower().split() if (word not in stopWords)] for document in documents]
	dictionary = corpora.Dictionary(texts)

	corpus = [dictionary.doc2bow(text) for text in texts]

	tfidf = models.TfidfModel(corpus)
	corpusTfidf = tfidf[corpus]

	numpyMatrix = gensim.matutils.corpus2dense(corpus, num_terms=20000)
	s = np.linalg.svd(numpyMatrix, full_matrices=False, compute_uv=False)

	'''
	plt.figure(figsize=(10,5))
	plt.hist(s, bins=100)
	plt.xlabel('Singular Values', fontsize=12)
	plt.show()
	'''

	lsi = models.LsiModel(corpusTfidf, id2word=dictionary, num_topics=30)
	index = similarities.MatrixSimilarity(lsi[corpusTfidf])
	pickle.dump(corpusTfidf,open('corpusTfidf.p','wb'))
	pickle.dump(lsi,open('lsi.p','wb'))
	pickle.dump(index,open('index.p','wb'))

else:
	corpusTfidf = pickle.load(open('corpusTfidf.p','rb'))
	lsi = pickle.load(open('lsi.p','rb'))
	index =  pickle.load(open('index.p','rb'))

df['similarity'] = 'unknown'
df['size_similar'] = 0
totalSims = []
threshold = 0.2
for i, doc in enumerate(corpusTfidf):
	vec_lsi = lsi[doc]
	sims = index[vec_lsi]
	totalSims = np.concatenate([totalSims,sims])
	similarity = []
	for j, x in enumerate(df.Name):
		if sims[j] > threshold:
			similarity.append((x,sims[j]))
	similarity = sorted(similarity, key=lambda item: -item[1])
	df = df.set_value(i,'similarity',similarity)
	df = df.set_value(i,'size_similar',len(similarity))


db_similarity = df[['Name','similarity']]

df1 = db_similarity
df2 = pd.read_csv('rating.txt', sep='\t')
threshold = 3
'''
df2 = pd.DataFrame(columns=('Name','rating'))


for i in range(len(df2)):
	if df2['Rating'][i] >= threshold:
		print('lel')
		df2.append(df2.ix[i])

print (df2)
'''
rec = []
for i in range(len(df1)):
	for j in range(len(df2)):
		if df1['Name'][i] == df2['Name'][j]:
			if df2['Rating'][j] >= threshold:
				similarity = df1['similarity'][i][1:]
				for tupl in similarity:
					rec.append((tupl[0], tupl[1]*df2['Rating'][j]))
				break

rec = sorted(rec, key=lambda item: -item[1])

print (rec[:10])
db_similarity.to_csv('final.txt', sep='\t')