import string
import textmining
import nltk
import re
import math
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

phrases = ['Today the aggies won!!!!!! Go aggies!', 'Texas A&M aggies have won today', 'the aggies lost last week', 'Find the Aggies news last week', 'An Aggie is a student at Texas A&M']

# Preprocessing
print("Preprocessing:")
for phrase in phrases:
	processed = ''.join(i for i in phrase if i not in string.punctuation)
	processed = processed.split()
	processed = ' '.join(i for i in processed if i.lower() not in stop_words or i.lower() == 'won')
	processed = processed.lower()
	processed = processed.split()
	processed = ' '.join(lemmatizer.lemmatize(i) for i in processed)
	processed = processed.split()
	processed = ' '.join(ps.stem(i) for i in processed)
	print(processed)
print

# Construct a term-document incidence matrix

doc1 = 'today aggi won go aggi'
doc2 = 'texa aggi won today'
doc3 = 'aggi lost last week'
doc4 = 'find aggi news last week'
doc5 = 'aggi student texa'

tdm = textmining.TermDocumentMatrix()

tdm.add_doc(doc1)
tdm.add_doc(doc2)
tdm.add_doc(doc3)
tdm.add_doc(doc4)
tdm.add_doc(doc5)

print("Term-document incidence matrix:")
for row in tdm.rows(cutoff=1):
	print(row)

print

# Implement TF-IDF on term-document incidence matrix

sent = """
today aggi won go aggi. 
texa aggi won today. 
aggi lost last week. 
find aggi news last week. 
aggi student texa.
"""

def remove_special(s):
	stripped = re.sub('[^\w\s]', '', s)
	stripped = re.sub('_', '', stripped)
	stripped = re.sub('\s+', ' ', stripped)
	stripped = stripped.strip()
	return stripped

def doc(sent):
	doc_info = []
	i = 0
	for s in sent:
		i += 1
		count = count_words(s)
		temp = {'doc' : i, 'length' : count}
		doc_info.append(temp)
	return doc_info

def count_words(s):
	count = 0
	words = word_tokenize(s)
	for word in words:
		count += 1
	return count

def create_dict(sent):
	i = 0
	freq = []
	for s in sent:	
		i += 1
		freq_dict = {}
		words = word_tokenize(s)
		for word in words:
			word = word.lower()
			if word in freq_dict:
				freq_dict[word] += 1
			else:
				freq_dict[word] = 1
			temp = {'doc' : i, 'freq_dict' : freq_dict}
		freq.append(temp)
	return freq

def computeTF(doc_info, freq):
	tf = []
	for temp in freq:
		id = temp['doc']
		for k in temp['freq_dict']:
			t = {'doc' : id, 'TF' : math.log10(1+temp['freq_dict'][k]), 'key' : k}
			tf.append(t)
	return tf

def computeIDF(doc_info, freq):
	idf = []
	count = 0
	for d in freq:
		count += 1
		for k in d['freq_dict'].keys():
			N = sum([k in temp['freq_dict'] for temp in freq])
			t = {'doc' : count, 'IDF' : math.log10(len(doc_info)/N), 'key' : k}
			idf.append(t)
	return idf

def computeTFIDF(TF, IDF):
	TFIDF = []
	for i in IDF:
		for j in TF:	
			if i['key'] == j['key'] and i['doc'] == j['doc']:
				t = {'doc' : i['doc'], 'TF-IDF' : i['IDF']*j['TF'], 'key' : j['key']}
		TFIDF.append(t)
	return TFIDF

sent_tok = sent_tokenize(sent)
text_clean = [remove_special(s) for s in sent_tok]
doc_info = doc(text_clean)

#create frequency table
freq_dict = create_dict(text_clean)

#get tf scores
tf_scores = computeTF(doc_info, freq_dict)

#get idf scores
idf_scores = computeIDF(doc_info, freq_dict)
tfidf = computeTFIDF(tf_scores, idf_scores)

print("TF-IDF:")
for i in range(0, len(tfidf)):
	print(tfidf[i]['doc'], tfidf[i]['key'], tfidf[i]['TF-IDF'])	

print


print("Cosine Similarity:")
doc1 = [0.09061905828945654, 0, 0.09061905828945654, 0.21041093737452468, 0, 0, 0, 0, 0, 0, 0]
doc2 = [0.09061905828945654, 0, 0.09061905828945654, 0, 0.09061905828945654, 0, 0, 0, 0, 0, 0]
doc3 = [0, 0, 0, 0, 0, 0.21041093737452468, 0.09061905828945654, 0.09061905828945654, 0, 0, 0]
doc4 = [0, 0, 0, 0, 0, 0, 0.09061905828945654, 0.09061905828945654, 0.21041093737452468, 0.21041093737452468, 0]
doc5 = [0, 0, 0, 0, 0.09061905828945654, 0, 0, 0, 0, 0, 0.21041093737452468]

docs = [doc1, doc2, doc3, doc4, doc5]

def cos(d1, d2):
	ans = 0	
	for i in range(0, len(d1)):
		ans += d1[i] * d2[i]
	return ans

def check(doc, docs):
	sim = []
	for d in range(0, len(docs)):
		sim.append(cos(doc, docs[d]))
	return sim

print(check(doc1, docs))
print(check(doc2, docs))
print(check(doc3, docs))
print(check(doc4, docs))
print(check(doc5, docs))

	
